"""GPU smoke: PYTHONPATH=src python benchmark/drop_check.py --videos VIDEOS --region REGION --weights WEIGHTS --model-source SOURCE --out OUT"""  # noqa: E501

from __future__ import annotations

import argparse
import json

from c2_check import add_inputs, setup

STEM = "Dashboard_user_id_13522_NoAudio_7"


def first_batch(bench):
    import torch

    from vitprune.pipeline.benchmark.inference import video_info
    from vitprune.pipeline.benchmark.settings import GEOMETRY
    from vitprune.pipeline.video import iter_clips

    videos = [video for video in bench.evaluated() if video.stem == STEM]
    if len(videos) != 1:
        raise ValueError(f"expected exactly one video named {STEM}")
    video = videos[0]
    starts = bench.clip_starts(video_info(video)["frames"])[:8]
    if len(starts) != 8:
        raise ValueError(f"{STEM}: need a complete scheduled batch of 8 clips")
    decoded = dict(iter_clips(video, GEOMETRY, 16, 2, set(starts)))
    if any(start not in decoded for start in starts):
        raise ValueError(f"{STEM}: incomplete decoded batch ({len(decoded)}/8)")
    return torch.stack([decoded[start] for start in starts]), starts


def observe(encoder, batch, device):
    import torch

    from vitprune.bench.latency import autocast

    block = encoder.backbone.blocks[2]
    original_run, original_forward = encoder._run, block.forward
    shapes = []
    hook = block.register_forward_pre_hook(lambda module, args: shapes.append(list(args[0].shape)))
    # SDPA normally drives the block directly, bypassing Module.__call__ and its hooks.
    block.forward = lambda x: original_run(block, x, None)
    encoder._run = lambda current, x, sizes: (
        current(x) if current is block else original_run(current, x, sizes))
    try:
        with torch.inference_mode(), autocast(device, "fp16"):
            embedding = encoder.encode(batch.to(device))
    finally:
        hook.remove()
        block.forward = original_forward
        encoder._run = original_run
    return embedding, shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    args = parser.parse_args()

    import torch

    from vitprune.core import FixedBudget, PrunedEncoder, PruningPlan
    from vitprune.pipeline.benchmark.settings import GEOMETRY

    bench, model, region = setup(args)
    batch, starts = first_batch(bench)
    plan = PruningPlan(prune_layer=2, budget=FixedBudget(1352), spatial_prior=region,
                       stages=((10, 1024), (20, 768)), attention="sdpa", discard_outside=True)
    encoder = PrunedEncoder(model, GEOMETRY, plan)
    embedding, shapes = observe(encoder, batch, bench.device)
    finite = bool(torch.isfinite(embedding).all())
    passed = shapes == [[8, 1352, 1408]] and finite
    result = {"video": STEM, "starts": starts, "shapes": shapes, "finite": finite,
              "embedding_max_abs": float(embedding.abs().max()) if finite else None,
              "gate": passed}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    print("PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
