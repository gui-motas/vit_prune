"""Check progressive region-separated merges on the 48 C2 dev clips (FP16 + SDPA)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from c2_check import add_inputs, sample_batches, setup

B2 = Path("/home/gsilv257/DAR-sys/vit_prune/benchmark/results/fp16-sdpa/b2")
PROGRESSIVE = "prog_1536_1024_768_mascara"


def load_baseline(bench, manifest, root):
    """Select saved B2 rows in the sampled clip order, refusing incomplete schedules."""
    import numpy as np

    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if any(run["settings"][key] != getattr(bench, key) for key in ("stride", "frame_step")):
        raise ValueError("B2 sampling settings differ from this check")
    videos, selected = {}, []
    for batch in manifest:
        name = batch["video"]
        if name not in videos:
            schedule = run["schedules"][name]
            starts = bench.clip_starts(schedule["frames"])
            path = root / "features" / PROGRESSIVE / f"{name}.npz"
            with np.load(path, allow_pickle=False) as data:
                features = data["feats"]
            if features.ndim != 2:
                raise ValueError(f"{name}: B2 feats must have shape [clips, dimensions]")
            if len(starts) != schedule["clips"] or len(features) != len(starts):
                raise ValueError(f"{name}: {len(features)} saved rows, {len(starts)} scheduled "
                                 f"starts, {schedule['clips']} recorded clips")
            if not np.isfinite(features).all():
                raise ValueError(f"{name}: non-finite B2 features")
            videos[name] = (features, {start: index for index, start in enumerate(starts)})
        features, positions = videos[name]
        try:
            rows = [positions[start] for start in batch["starts"]]
        except KeyError as error:
            raise ValueError(f"{name}: sampled start {error.args[0]} absent from B2") from error
        selected.append(features[rows])
    return selected


def difference(actual, reference):
    import torch
    import torch.nn.functional as F

    if actual.shape != reference.shape or not torch.isfinite(actual).all() or not (
            torch.isfinite(reference).all()):
        return {"max_abs_delta": None, "min_cosine": None}
    return {"max_abs_delta": float((actual - reference).abs().max()),
            "min_cosine": float(F.cosine_similarity(actual, reference, dim=-1).min())}


def write_result(out, result):
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                    encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["control"]["passed"]:
        print("CONTROLE FALHOU", flush=True)
        return 2
    print(result["decision"], flush=True)
    return 0


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

    import numpy as np
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
    try:
        reference = load_baseline(bench, manifest, B2)
        if len(reference) != len(batches) or any(
                len(saved) != len(batch) for saved, batch in zip(reference, batches, strict=True)):
            raise ValueError("B2 samples do not match the decoded batches")
    except (OSError, ValueError, KeyError) as error:
        return write_result(args.out, {"batches": manifest, "precision": "fp16",
                                      "attention": "sdpa", "control": {"passed": False,
                                      "baseline": str(B2), "error": str(error)}})
    plans = {name: PruningPlan(prune_layer=2, budget=FixedBudget(1536), spatial_prior=region,
                               stages=((10, 1024), (20, 768)), attention="sdpa",
                               separate_region=separated)
             for name, separated in (("P", False), ("P-sep", True))}
    encoders = {name: PrunedEncoder(model, GEOMETRY, plan) for name, plan in plans.items()}
    totals = {name: {str(stage): {"cross": 0, "total": 0, "forced": 0}
                     for stage in (2, 10, 20)} for name in plans}
    embeddings = {name: [] for name in plans}
    with torch.inference_mode():
        for batch in batches:
            clips = batch.to(bench.device)
            with autocast(bench.device, "fp16"):
                for name, encoder in encoders.items():
                    embedding, counts = diagnose(encoder, clips, region)
                    embeddings[name].append(embedding.float().cpu())
                    for stage, row in counts.items():
                        for key, value in row.items():
                            totals[name][stage][key] += value
    for arm in totals.values():
        for row in arm.values():
            row["fraction"] = row["cross"] / row["total"] if row["total"] else 0.0
    p = torch.cat(embeddings["P"])
    sep = torch.cat(embeddings["P-sep"])
    baseline = torch.from_numpy(np.concatenate(reference)).float()
    control = difference(p, baseline)
    control["passed"] = (control["max_abs_delta"] is not None
                         and control["max_abs_delta"] <= 1e-2
                         and control["min_cosine"] >= .9999)
    control["baseline"] = str(B2)
    control["max_abs_delta_limit"], control["min_cosine_limit"] = 1e-2, .9999
    result = {"batches": manifest, "precision": "fp16", "attention": "sdpa",
              "control": control, "p_vs_p_sep": difference(p, sep), "stages": totals}
    if control["passed"]:
        if result["p_vs_p_sep"]["max_abs_delta"] is None:
            raise ValueError("P-sep produced non-finite embeddings")
        if any(row["cross"] != row["forced"] for row in totals["P-sep"].values()):
            raise ValueError("P-sep has a cross-boundary merge that was not forced")
        result["decision"] = ("ADIAR" if all(row["fraction"] < .01
                                             for row in totals["P"].values()) else "PROSSEGUIR")
    return write_result(args.out, result)


if __name__ == "__main__":
    raise SystemExit(main())
