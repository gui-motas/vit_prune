"""Run: PYTHONPATH=src python benchmark/c2_check.py --videos VIDEOS --region REGION --weights WEIGHTS --model-source SOURCE --out OUT"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def add_inputs(parser):
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--region", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output directory")


def setup(args):
    import torch

    from vitprune.pipeline.benchmark.inference import load_model
    from vitprune.pipeline.benchmark.settings import GEOMETRY, Benchmark
    from vitprune.pipeline.region import read_region

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these measurements")
    bench = Benchmark(videos=args.videos, out=args.out, policy=None, region=args.region,
                      weights=args.weights, model_source=args.model_source, masked=(1024,))
    model = load_model(bench)
    region = read_region(args.region, GEOMETRY)
    return bench, model, region


def sample_starts(starts):
    middle = (len(starts) // 8 // 2) * 8
    offsets = [0] if middle == 0 else [0, middle]
    return [starts[offset:offset + 8] for offset in offsets]


def sample_batches(bench):
    import torch

    from vitprune.pipeline.benchmark.inference import video_info
    from vitprune.pipeline.benchmark.settings import GEOMETRY
    from vitprune.pipeline.video import iter_clips

    batches, manifest = [], []
    for video in bench.evaluated():
        groups = sample_starts(bench.clip_starts(video_info(video)["frames"]))
        by_start = {start: True for group in groups for start in group}
        decoded = dict(iter_clips(video, GEOMETRY, 16, 2, by_start))
        for starts in groups:
            available = [start for start in starts if start in decoded]
            if len(available) != 8:
                print(f"{video.stem}: skipped incomplete batch ({len(available)}/8)", flush=True)
                continue
            batches.append(torch.stack([decoded[start] for start in available]))
            manifest.append({"video": video.stem, "starts": available})
        count = sum(m["video"] == video.stem for m in manifest)
        print(f"{video.stem}: {count} batches", flush=True)
    if not batches:
        raise ValueError("no complete decoded batch of eight clips")
    return batches, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile

    from vitprune.bench.latency import autocast
    from vitprune.core import FixedBudget, PrunedEncoder, PruningPlan
    from vitprune.pipeline.benchmark.settings import GEOMETRY

    bench, model, region = setup(args)
    batches, manifest = sample_batches(bench)
    configs = [("fp32_model", "off", "model"), ("fp16_model", "fp16", "model"),
               ("fp16_sdpa", "fp16", "sdpa")]
    results, reference, fixed_reference = {}, None, None
    with torch.inference_mode():
        for arm in ("denso", "fixo_1024_mascara"):
            for name, precision, attention in configs:
                plan = (PruningPlan.dense(attention=attention) if arm == "denso" else
                        PruningPlan(prune_layer=2, budget=FixedBudget(1024), spatial_prior=region,
                                    attention=attention))
                encoder = PrunedEncoder(model, GEOMETRY, plan)
                embeddings, probabilities, selections = [], [], []
                for batch in batches:
                    x = batch.to(bench.device)
                    with autocast(bench.device, precision):
                        embedding = encoder.encode(x)
                        probability = model.head(embedding).float().softmax(dim=-1)
                        if arm != "denso":
                            selections.append(encoder.select(x).kept.cpu())
                    embeddings.append(embedding.float().cpu())
                    probabilities.append(probability.cpu())
                embeddings, probabilities = torch.cat(embeddings), torch.cat(probabilities)
                if reference is None:
                    reference = (embeddings, probabilities)
                if selections:
                    selections = torch.cat(selections)
                    if fixed_reference is None:
                        fixed_reference = selections
                cosine = F.cosine_similarity(embeddings, reference[0], dim=-1)
                finite = bool(torch.isfinite(embeddings).all()
                              and torch.isfinite(probabilities).all()
                              and torch.isfinite(cosine).all())
                row = {"finite": finite,
                       "cosine_p10": float(torch.quantile(cosine, .1)) if finite else None,
                       "cosine_p50": float(torch.quantile(cosine, .5)) if finite else None,
                       "cosine_min": float(cosine.min()) if finite else None,
                       "embedding_max_abs_delta": float((embeddings - reference[0]).abs().max())
                       if finite else None,
                       "probability_max_abs_delta":
                       float((probabilities - reference[1]).abs().max()) if finite else None,
                       "top1_equal_fraction": float((probabilities.argmax(-1)
                                                     == reference[1].argmax(-1)).float().mean())
                       if finite else None,
                       "kept_identical_fraction": float((selections == fixed_reference)
                                                        .flatten(1).all(dim=1).float().mean())
                       if arm != "denso" else None}
                results[f"{arm}/{name}"] = row
                print(f"{arm}/{name}: {row}", flush=True)
        kernels = {}
        for name, precision in (("fp16_sdpa", "fp16"), ("fp32_sdpa", "off")):
            encoder = PrunedEncoder(model, GEOMETRY, PruningPlan.dense(attention="sdpa"))
            x = batches[0].to(bench.device)
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                with autocast(bench.device, precision):
                    encoder.encode(x)
                torch.cuda.synchronize()
            kernels[name] = sorted({event.key for event in prof.key_averages()
                                    if re.search(r"flash|fmha|efficient|attention|softmax|cutlass",
                                                 event.key, re.IGNORECASE)})
            print(f"{name} profiler: {kernels[name]}", flush=True)
    p10 = results["denso/fp16_sdpa"]["cosine_p10"]
    passed = all(row["finite"] for row in results.values()) and p10 is not None and p10 >= .999
    result = {"batches": manifest, "configurations": results, "kernels": kernels, "gate": passed}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    columns = ("cosine_p10", "cosine_p50", "cosine_min", "embedding_max_abs_delta",
               "probability_max_abs_delta", "top1_equal_fraction", "kept_identical_fraction",
               "finite")
    print("| Configuration | " + " | ".join(columns) + " |", flush=True)
    for name, row in results.items():
        values = [f"{v:.6g}" if isinstance(v, float) else str(v) for v in
                  (row[column] for column in columns)]
        print(f"| {name} | " + " | ".join(values) + " |", flush=True)
    print("GATE PASS" if passed else "GATE FAIL", flush=True)


if __name__ == "__main__":
    main()
