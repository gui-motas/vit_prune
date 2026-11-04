"""Is it the choice of tokens or their number? A diagnostic of the selection criterion.

At the same budget, every clip of one video of a finished policy study is encoded again with
other criteria for which tokens to keep, and compared through the AMA with the run's own
dense and fixed variants, whose features are linked rather than recomputed:

* ``mascara_K``: the driver region as a spatial prior, as the mask study applies it;
* ``mascara_sem_carimbo_K``: the same, with the burned-in date and time left out as well;
* ``aleatorio_K``: tokens kept at random, no prior: the floor any criterion has to beat.

Writes a run folder that ``vitprune benchmark ama`` and ``benchmark report`` take; a
variant already written is kept, so an interrupted run resumes. Exploratory, on a selection
video: nothing here tunes a parameter.

    python benchmark/qabp_selection.py --run benchmark/results/qabp/selecao \\
        --video Dashboard_user_id_28557_NoAudio_5 --region region-novo.json \\
        --weights W --model-source S --out benchmark/results/qabp/criterio
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vitprune.core import ClipGeometry, PrunedEncoder, PruningPlan
from vitprune.core.policy import FixedBudget
from vitprune.core.priors import box_prior
from vitprune.pipeline.checkpoint import disable_activation_checkpointing, load_videomae
from vitprune.pipeline.region import read_region
from vitprune.pipeline.video import find_videos, iter_clips

#: The date and time burned into the AI City Dashboard footage, as fractions of the frame.
TIMESTAMP = (0.62, 0.94, 0.92, 0.98)
CRITERIA = ("mascara", "mascara_sem_carimbo", "aleatorio")
LINKED = ("denso", "fixo_512", "fixo_1024")


class RandomSelection(PrunedEncoder):
    """Keeps tokens at random, the same draw for the same clips in the same order."""

    def __init__(self, *args, seed: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generator = torch.Generator().manual_seed(seed)

    def _with_prior(self, base: torch.Tensor) -> torch.Tensor:
        return torch.rand(base.shape, generator=self.generator).to(base.device)


def criteria(model, geometry: ClipGeometry, region: torch.Tensor, levels, cut: int) -> dict:
    """``{name: encoder}`` for every criterion and level."""
    stamp = box_prior(geometry.grid_size, [TIMESTAMP])
    no_stamp = region * (1 - stamp)
    assert no_stamp.sum() < region.sum(), "the timestamp lies outside the region"
    found = {}
    for k in levels:
        budget = FixedBudget(k)
        found[f"mascara_{k}"] = PrunedEncoder(
            model, geometry, PruningPlan(prune_layer=cut, budget=budget, spatial_prior=region))
        found[f"mascara_sem_carimbo_{k}"] = PrunedEncoder(
            model, geometry, PruningPlan(prune_layer=cut, budget=budget, spatial_prior=no_stamp))
        found[f"aleatorio_{k}"] = RandomSelection(
            model, geometry, PruningPlan(prune_layer=cut, budget=budget))
    return found


@torch.no_grad()
def encode_video(model, encoder, path: Path, geometry, stride: int, step: int, batch: int,
                 device: str):
    """Every clip of the video: embeddings, the head's probabilities, starts, seconds."""
    feats, probs, starts, seconds = [], [], [], []
    pending = []

    def flush():
        clips = torch.stack([clip for _, clip in pending]).to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        began = time.perf_counter()
        embedding = encoder.encode(clips)
        probabilities = model.head(embedding).float().softmax(dim=-1)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        spent = (time.perf_counter() - began) / len(pending)
        feats.append(embedding.float().cpu().numpy())
        probs.append(probabilities.cpu().numpy())
        starts.extend(start for start, _ in pending)
        seconds.extend([spent] * len(pending))
        pending.clear()

    for item in iter_clips(path, geometry, stride, step):
        pending.append(item)
        if len(pending) == batch:
            flush()
    if pending:
        flush()
    return np.concatenate(feats), np.concatenate(probs), starts, seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="the policy study the video comes from")
    parser.add_argument("--video", required=True, help="the stem of one of its videos")
    parser.add_argument("--region", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--criteria", nargs="+", choices=CRITERIA, default=list(CRITERIA))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source, out, stem = Path(args.run), Path(args.out), args.video
    run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    settings = run["settings"]
    stride, step, batch = settings["stride"], settings["frame_step"], settings["batch"]
    geometry = ClipGeometry.videomae_giant()
    region = read_region(args.region, geometry)
    paths = {path.stem: path for path in find_videos(run["videos"])}

    model = disable_activation_checkpointing(
        load_videomae(args.weights, args.model_source, geometry, device=args.device))
    encoders = {name: encoder for name, encoder in
                criteria(model, geometry, region, args.levels, settings["prune_layer"]).items()
                if name.rsplit("_", 1)[0] in args.criteria}
    variants = [*LINKED, *encoders]

    for name in (*LINKED, "frames", "selection"):
        target = out / "features" / name / f"{stem}.npz" if name in LINKED else out / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            origin = (source / "features" / name / f"{stem}.npz" if name in LINKED
                      else source / name)
            target.symlink_to(origin.resolve())
    dense = np.load(out / "features" / "denso" / f"{stem}.npz")["feats"]
    record = {**run, "schedules": {stem: run["schedules"][stem]},
              "settings": {**settings, "variants": variants, "evaluate": [stem],
                           "fixed": [512, 1024], "proportional": []},
              "criterion_study": {"source": str(source), "region": args.region,
                                  "timestamp": TIMESTAMP, "random_seed": 0}}
    (out / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    rows = out / "clips.jsonl"
    if not rows.exists():
        with (source / "clips.jsonl").open(encoding="utf-8") as stream:
            rows.write_text("".join(line for line in stream if json.loads(line)["video"] == stem
                                    and json.loads(line)["variant"] in LINKED),
                            encoding="utf-8")
    fps = run["schedules"][stem]["fps"]

    for name, encoder in encoders.items():
        target = out / "features" / name / f"{stem}.npz"
        if target.exists():
            print(f"{name}: done before", flush=True)
            continue
        began = time.time()
        feats, probs, starts, seconds = encode_video(model, encoder, paths[stem], geometry,
                                                     stride, step, batch, args.device)
        assert len(feats) == len(dense), f"{name}: {len(feats)} clips, dense has {len(dense)}"
        reference = torch.from_numpy(dense).double()
        mine = torch.from_numpy(feats).double()
        cosines = F.cosine_similarity(mine, reference, dim=-1).tolist()
        relative = ((mine - reference).norm(dim=-1) / reference.norm(dim=-1)).tolist()
        tokens = encoder.plan.budget.tokens
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, feats=feats, a_prob=probs)
        with rows.open("a", encoding="utf-8") as stream:
            for i, start in enumerate(starts):
                stream.write(json.dumps({
                    "video": stem, "start_frame": start, "start_s": round(start / fps, 3),
                    "variant": name, "tokens": tokens, "regime": 0, "inside_region": None,
                    "cosine_to_dense": cosines[i], "rel_l2_to_dense": relative[i],
                    "predicted_risk": None, "reason": None, "seconds": round(seconds[i], 4),
                }) + "\n")
        print(f"{name}: {len(feats)} clips in {time.time() - began:.0f}s, cosine P50 with the "
              f"dense {np.median(cosines):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
