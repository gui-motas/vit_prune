"""Run: PYTHONPATH=src python benchmark/arms_latency.py --videos VIDEOS --region REGION --weights WEIGHTS --model-source SOURCE --out OUT --precision fp16 --attention sdpa --arms denso fixo_1024_mascara prog:1536,10:1024,20:768"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import time

from c2_check import add_inputs, sample_batches, setup


def make_plans(arms, region, attention):
    from vitprune.core import FixedBudget, PruningPlan
    from vitprune.pipeline.benchmark.settings import GEOMETRY

    if len(set(arms)) != len(arms):
        raise ValueError("arm names must be unique")
    plans = {}
    for arm in arms:
        if arm == "denso":
            plan = PruningPlan.dense(attention=attention)
        elif arm == "fixo_1024_mascara":
            plan = PruningPlan(prune_layer=2, budget=FixedBudget(1024), spatial_prior=region,
                               attention=attention)
        elif arm.startswith(("prog:", "prog-sep:")):
            parts = arm.split(":", 1)[1].split(",")
            try:
                initial = int(parts[0])
                stages = tuple(tuple(map(int, part.split(":"))) for part in parts[1:])
            except ValueError as error:
                raise ValueError(f"invalid progressive arm: {arm}") from error
            if not stages:
                raise ValueError(f"progressive arm needs later stages: {arm}")
            plan = PruningPlan(prune_layer=2, budget=FixedBudget(initial), spatial_prior=region,
                               attention=attention, stages=stages,
                               separate_region=arm.startswith("prog-sep:"))
        else:
            raise ValueError(f"unknown arm: {arm}")
        plan.validate_for(GEOMETRY)
        plans[arm] = plan
    return plans


def mean_tokens(plan, geometry):
    if not plan.prunes:
        return float(geometry.num_tokens)
    stages = ((0, geometry.num_tokens), (plan.prune_layer, plan.budget.tokens), *plan.stages,
              (geometry.depth, 0))
    return sum((end[0] - start[0]) * start[1]
               for start, end in zip(stages, stages[1:], strict=False)) / geometry.depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    parser.add_argument("--precision", choices=("off", "fp16"), default="off")
    parser.add_argument("--attention", choices=("model", "sdpa"), default="model")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--passes", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 0 or args.passes < 1:
        parser.error("--warmup must be non-negative and --passes must be positive")

    import numpy as np
    import torch

    from vitprune.bench.latency import autocast
    from vitprune.core import PrunedEncoder
    from vitprune.pipeline.benchmark.settings import GEOMETRY

    bench, model, region = setup(args)
    plans = make_plans(args.arms, region, args.attention)
    encoders = {name: PrunedEncoder(model, GEOMETRY, plan) for name, plan in plans.items()}
    batches, manifest = sample_batches(bench)
    times = {name: [] for name in plans}
    with torch.inference_mode():
        for turn in range(args.warmup + args.passes):
            warm = turn < args.warmup
            names = args.arms[turn % len(args.arms):] + args.arms[:turn % len(args.arms)]
            for batch in batches:
                x = batch.to(bench.device)
                for name in names:
                    if not warm:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                    with autocast(bench.device, args.precision):
                        embedding = encoders[name].encode(x)
                        model.head(embedding)
                    if not warm:
                        torch.cuda.synchronize()
                        times[name].append((time.perf_counter() - started) * 1000 / 8)
            print(f"{'warmup' if warm else 'pass'} {turn + 1}: {names}", flush=True)
    result = {}
    for name, samples in times.items():
        result[name] = {"p50_ms": float(np.percentile(samples, 50)),
                        "p95_ms": float(np.percentile(samples, 95)),
                        "mean_ms": float(np.mean(samples)), "batches": len(samples),
                        "mean_tokens_per_block": mean_tokens(plans[name], GEOMETRY)}
    baseline = result.get("fixo_1024_mascara")
    for name, row in result.items():
        row["ratio_vs_fixo_1024_mascara"] = (
            {key: row[key] / baseline[key] for key in ("p50_ms", "p95_ms", "mean_ms")}
            if baseline else None)
        print(f"{name}: {row}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    output = {"precision": args.precision, "attention": args.attention, "warmup": args.warmup,
              "passes": args.passes, "batches": manifest, "arms": result}
    (args.out / "latency.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8")
    print(f"wrote {args.out / 'latency.json'}", flush=True)


if __name__ == "__main__":
    main()
