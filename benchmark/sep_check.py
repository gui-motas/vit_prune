"""Check progressive region-separated merges on the 48 C2 dev clips (FP16 + SDPA)."""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch

from c2_check import add_inputs, sample_batches, setup


def count_cross(source, destination, candidates):
    """Count crossings and those with no same-group destination available."""
    cross = source != destination
    available = (source.unsqueeze(-1) == candidates.unsqueeze(-2)).any(dim=-1)
    return {"cross": int(cross.sum()), "total": source.numel(),
            "forced": int((cross & ~available).sum())}


def diagnose(encoder, clips, prior):
    """Recompute the actual matches from the tensors seen by each merge."""
    import torch

    import vitprune.core.encoder as module
    from vitprune.core.merging import _bipartite_matches, _nearest_kept_token, _take

    cut_merge, stage_merge = module.merge_tokens, module.bipartite_merge
    region = (prior.to(clips.device) > 0)
    tracked = None
    stages = iter(block for block, _ in encoder.plan.stages)
    counts = {}

    def cut(tokens, sizes, scores, keep, groups=None):
        nonlocal tracked
        ranking = scores.argsort(dim=-1, descending=True)
        kept = ranking[..., :keep].sort(dim=-1).values
        removed = ranking[..., keep:]
        source, candidates = region[removed], region[kept]
        pair = (source, candidates) if groups is not None else None
        dest = _nearest_kept_token(_take(tokens, removed), _take(tokens, kept), pair)
        counts[str(encoder.plan.prune_layer)] = count_cross(
            source, candidates.gather(-1, dest), candidates)
        tracked = candidates.reshape(tokens.shape[0], -1)
        return cut_merge(tokens, sizes, scores, keep, groups)

    def stage(tokens, sizes, metric, keep, groups=None):
        nonlocal tracked
        if tracked is None:
            tracked = region.repeat(encoder.geometry.temporal_slices).expand(tokens.shape[0], -1)
            counts[str(encoder.plan.prune_layer)] = {"cross": 0, "total": 0, "forced": 0}
        unmerged, source_at, destination_at = _bipartite_matches(metric, keep, groups)
        a, b = tracked[:, ::2], tracked[:, 1::2]
        source = a.gather(1, source_at.squeeze(-1))
        destination = b.gather(1, destination_at.squeeze(-1))
        counts[str(next(stages))] = count_cross(source, destination, b)
        tracked = torch.cat((a.gather(1, unmerged.squeeze(-1)), b), dim=1)
        return stage_merge(tokens, sizes, metric, keep, groups)

    with patch.object(module, "merge_tokens", cut), patch.object(module, "bipartite_merge", stage):
        embedding = encoder.encode(clips)
    return embedding, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    args = parser.parse_args()

    import torch

    from vitprune.bench.latency import autocast
    from vitprune.core import FixedBudget, PrunedEncoder, PruningPlan
    from vitprune.pipeline.benchmark.settings import GEOMETRY

    bench, model, region = setup(args)
    bench.evaluate = ("Dashboard_user_id_13522_NoAudio_7", "Dashboard_user_id_31903_NoAudio_7",
                      "Dashboard_user_id_33682_NoAudio_5")
    batches, manifest = sample_batches(bench)
    if len(batches) != 6:
        raise ValueError("need two batches from each of the three dev videos")
    plans = {name: PruningPlan(prune_layer=2, budget=FixedBudget(1536), spatial_prior=region,
                               stages=((10, 1024), (20, 768)), attention="sdpa",
                               separate_region=separated)
             for name, separated in (("P", False), ("P-sep", True))}
    encoders = {name: PrunedEncoder(model, GEOMETRY, plan) for name, plan in plans.items()}
    totals = {name: {str(stage): {"cross": 0, "total": 0, "forced": 0}
                     for stage in (2, 10, 20)} for name in plans}
    max_delta = 0.0
    with torch.inference_mode():
        for batch in batches:
            clips = batch.to("cuda")
            with autocast("cuda", "fp16"):
                reference = encoders["P"].encode(clips)
                if not torch.isfinite(reference).all():
                    raise ValueError("P control produced non-finite embeddings")
                for name, encoder in encoders.items():
                    embedding, counts = diagnose(encoder, clips, region)
                    if not torch.isfinite(embedding).all():
                        raise ValueError(f"{name} produced non-finite embeddings")
                    if name == "P":
                        max_delta = max(max_delta, float((embedding.float() - reference.float())
                                                         .abs().max()))
                    for stage, row in counts.items():
                        for key, value in row.items():
                            totals[name][stage][key] += value
    for arm in totals.values():
        for row in arm.values():
            row["fraction"] = row["cross"] / row["total"] if row["total"] else 0.0
    if any(row["cross"] != row["forced"] for row in totals["P-sep"].values()):
        raise ValueError("P-sep has a cross-boundary merge that was not forced")
    decision = ("ADIAR" if all(row["fraction"] < .01 for row in totals["P"].values())
                else "PROSSEGUIR")
    result = {"batches": manifest, "precision": "fp16", "attention": "sdpa",
              "control": {"comparison": "default P encode vs P encode with groups=None explicitly",
                          "max_abs_delta": max_delta, "tolerance": 1e-5},
              "stages": totals, "decision": decision}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(decision, flush=True)
    if max_delta > 1e-5:
        raise AssertionError(f"P control max|Δ| {max_delta} exceeds 1e-5")


if __name__ == "__main__":
    main()
