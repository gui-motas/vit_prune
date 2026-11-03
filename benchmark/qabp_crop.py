"""Does the driver crop change what pruning does to the embedding?

A diagnostic outside the pipeline. The AMA and the VideoMAEv2 fine-tuning saw AI City's
crop_videos, each video cropped to the driver's largest box; the pipeline reads the whole
frame. On clips of the P0 panel, or on any given, this runs every level against its own
dense path twice: on the whole frame, exactly as the pipeline reads it, and on the driver's
crop that the region's calibration found for that video by AI City's rule. Nothing else
changes: the same weights, cut, selection, merge and precision. For a clip of the P0 panel
the whole-frame result is set beside the P0's, measured on the GPU, which shows what the
device changes.

    python benchmark/qabp_crop.py --records benchmark/results/qabp/p0/records \\
        --videos benchmark/results/qabp/videos/p0 --region region-novo.json \\
        --weights W --model-source S --out benchmark/results/qabp/recorte \\
        [--device cpu --threads 6 --levels 512 --clips 13522_NoAudio_7@0 ...]

Writes OUT/recorte.jsonl, one line per clip, and resumes after an interruption.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from vitprune.core import ClipGeometry, PrunedEncoder, PruningPlan
from vitprune.pipeline.checkpoint import disable_activation_checkpointing, load_videomae
from vitprune.pipeline.quality import read_records
from vitprune.pipeline.video import _normalise, clip_span, find_videos, iter_clips, read_frames

STRIDE, FRAME_STEP, CUT = 16, 2, 2


def read_clip(path: Path, start: int, geometry: ClipGeometry, box=None) -> torch.Tensor:
    """The clip at ``start`` as the pipeline builds it, cropped first to ``box``
    (``x1, y1, x2, y2`` as fractions of the frame) when given."""
    span = clip_span(geometry, FRAME_STEP)
    frames = list(islice(read_frames(path, start), span))[::FRAME_STEP]
    if box is not None:
        height, width = frames[0].shape[:2]
        x1, y1, x2, y2 = box
        frames = [f[round(y1 * height):round(y2 * height), round(x1 * width):round(x2 * width)]
                  for f in frames]
    return _normalise(np.stack(frames), geometry.image_size).permute(1, 0, 2, 3)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1).item()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="the P0 records folder")
    parser.add_argument("--videos", required=True)
    parser.add_argument("--region", required=True, help="region JSON with its calibration")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads; 0 leaves torch's")
    parser.add_argument("--levels", type=int, nargs="+", default=[512, 1024, 1536])
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3, 10, 20, 30, 40])
    parser.add_argument("--clips", nargs="*", help="STEM@START; every P0 clip by default")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    records = {(r["video"], r["start"]): r for r in read_records(args.records)[2]}
    wanted = ([(f"Dashboard_user_id_{c.split('@')[0]}", int(c.split("@")[1])) for c in args.clips]
              if args.clips else sorted(records))
    calibration = json.loads(Path(args.region).read_text(encoding="utf-8"))["calibration"]
    boxes = {Path(v["source"]["path"]).stem: v["crop"] for v in calibration["videos"]}
    paths = {path.stem: path for path in find_videos(args.videos)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "recorte.jsonl"
    done = ({(r["video"], r["start"]) for r in map(json.loads, target.read_text().splitlines())}
            if target.exists() else set())

    geometry = ClipGeometry.videomae_giant()
    model = disable_activation_checkpointing(
        load_videomae(args.weights, args.model_source, geometry, device=args.device))
    encoder = PrunedEncoder(model, geometry, PruningPlan(prune_layer=CUT))

    for stem, start in wanted:
        if (stem, start) in done:
            continue
        whole = read_clip(paths[stem], start, geometry)
        _, reference = next(iter_clips(paths[stem], geometry, STRIDE, FRAME_STEP, [start]))
        assert torch.equal(whole, reference), "the whole-frame clip differs from the pipeline's"
        crop = read_clip(paths[stem], start, geometry, boxes[stem])

        began = time.perf_counter()
        found = encoder.encode_levels(torch.stack([whole, crop]).to(args.device), args.levels,
                                      args.depths)
        dense = geometry.num_tokens
        row = {"video": stem, "start": start, "box": boxes[stem], "device": args.device,
               "seconds": round(time.perf_counter() - began, 1), "inputs": {}}
        for i, name in enumerate(("inteiro", "recorte")):
            row["inputs"][name] = {str(level): {
                "cos": cosine(found.embeddings[level][i], found.embeddings[dense][i]),
                "stages": {str(d): cosine(found.stages[level][d][i], found.stages[dense][d][i])
                           for d in args.depths}} for level in args.levels}
        if (stem, start) in records:
            row["p0_cos"] = {str(level): 1.0 - records[(stem, start)]["levels"][str(level)]["d_cos"]
                             for level in args.levels}
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        for level in args.levels:
            key = str(level)
            p0 = row.get("p0_cos", {}).get(key)
            print(f"{stem}@{start} {level}: whole frame {row['inputs']['inteiro'][key]['cos']:+.3f}"
                  + ("" if p0 is None else f" (P0 on the GPU {p0:+.3f})")
                  + f", crop {row['inputs']['recorte'][key]['cos']:+.3f}  [{row['seconds']} s]",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
