"""Does a signal a budget could read at the cut predict what pruning costs the task?

The quality-aware budget needs a per-clip risk. The AMA's mAP is a measure of actions over
whole videos, not of clips, so it cannot be that risk. What a clip does have, when its
window lies inside one annotated action, is a class: and against it, what the pruned head
lost relative to the dense one. That loss is a candidate proxy of the harm to the task, to
be checked against the actions the AMA keeps or loses, never assumed to be it.

On a finished policy study with annotations (``vitprune benchmark ama --labels``):

* :func:`collect_cut_features` runs the blocks up to the cut on every clip of the run and
  records what a budget sees there -- motion, redundancy and the probe;
* :func:`analyse` measures, per pruned variant, the head's loss on every clip of one
  action, its relation to each signal (and to the embedding drift, which is not available
  online), what the probe adds beyond motion and redundancy, and which annotated actions
  the AMA lost against the dense run.

Everything here is exploratory: the clips of a video are not independent, the AMA saw these
videos in training, and nothing tunes a parameter. Writes under ``RUN/proxy/``:
``cut_features.jsonl``, ``proxy.json``, ``clips.csv`` and ``actions.csv``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from vitprune.bench.latency import autocast
from vitprune.bench.quality import auc, partial_spearman, permutation_p, spearman
from vitprune.core.compressibility import PROBE_CELLS, PROBE_TOP
from vitprune.pipeline.benchmark import settings
from vitprune.pipeline.benchmark.settings import RunRecord
from vitprune.pipeline.quality import driver_of
from vitprune.pipeline.video import find_videos, iter_clips

FOLDER = "proxy"
CUT = "cut_features.jsonl"
#: The signals compared; ``drift`` is measured after the last block, so no budget can read
#: it -- it is there to see whether the embedding's distance tracks the task at all.
SIGNALS = ("probe_p90", "probe_mean", "g", "motion", "redundancy", "drift")
LOSSES = ("delta_ce", "delta_p")


def clip_labels(starts: Sequence[float], span: float,
                actions: Sequence[tuple[float, float, int]]) -> list[int | None]:
    """The class of each clip whose window lies wholly inside exactly one annotated action and
    touches no other; ``None`` for background, a window across a boundary, or two actions."""
    labels = []
    for start in starts:
        end = start + span
        inside = [label for first, last, label in actions if first <= start and end <= last]
        touching = [label for first, last, label in actions if first < end and start < last]
        labels.append(inside[0] if len(inside) == 1 and len(touching) == 1 else None)
    return labels


def functional_losses(dense: np.ndarray, pruned: np.ndarray,
                      labels: Sequence[int]) -> dict[str, np.ndarray]:
    """Per labelled clip: the probability the pruned head gives the true class and its margin
    over the next class, how far it fell from the dense head's (``delta_p``, never negative),
    the rise of the cross-entropy (``delta_ce``, never negative), and whether a right dense
    answer turned wrong."""
    labels = np.asarray(labels, dtype=int)
    rows = np.arange(len(labels))
    true_dense, true_pruned = dense[rows, labels], pruned[rows, labels]
    others = pruned.copy()
    others[rows, labels] = -np.inf
    tiny = 1e-12
    return {
        "p_true": true_pruned,
        "p_true_dense": true_dense,
        "margin": true_pruned - others.max(axis=1),
        "delta_p": np.maximum(0.0, true_dense - true_pruned),
        "delta_ce": np.maximum(0.0, np.log(true_dense + tiny) - np.log(true_pruned + tiny)),
        "flip": (dense.argmax(axis=1) == labels) & (pruned.argmax(axis=1) != labels),
    }


@torch.no_grad()
def collect_cut_features(encoder, out: str | Path, device: str = "cuda", precision: str = "off",
                         batch: int = 8, cells: tuple[int, int] = PROBE_CELLS,
                         top: int = PROBE_TOP, progress=print) -> Path:
    """The signals at the cut of every clip of a finished run, video by video, resumable.

    ``encoder``'s plan must be the run's cut and selection; only its blocks up to the cut run.
    """
    out = Path(out)
    record = RunRecord.read(out)
    rows = _rows(out)
    target = out / FOLDER / CUT
    target.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["video"] for line in target.read_text("utf-8").splitlines()} \
        if target.exists() else set()
    paths = {path.stem: path for path in find_videos(record.run["videos"])}
    stride, step = record.settings["stride"], record.settings["frame_step"]
    for stem in record.stems:
        if stem in done:
            progress(f"[proxy] {stem}: done before")
            continue
        starts = sorted(r["start_frame"] for r in rows if r["video"] == stem
                        and r["variant"] == "denso")
        found, pending = [], []
        for item in iter_clips(paths[stem], encoder.geometry, stride, step, starts):
            pending.append(item)
            if len(pending) == batch:
                found += _cut(encoder, stem, pending, device, precision, cells, top)
                pending = []
        if pending:
            found += _cut(encoder, stem, pending, device, precision, cells, top)
        with target.open("a", encoding="utf-8") as stream:
            stream.write("".join(json.dumps(row) + "\n" for row in found))
        progress(f"[proxy] {stem}: {len(found)} clips")
    return target


def _cut(encoder, stem: str, pending: list, device: str, precision: str,
         cells: tuple[int, int], top: int) -> list[dict]:
    """The signals at the cut of a batch of ``(start, clip)``."""
    clips = torch.stack([clip for _, clip in pending]).to(device)
    with autocast(device, precision):
        features = encoder.features(clips, cells, top)
    columns = [value.tolist() for value in features]
    return [{"video": stem, "start": start,
             **{name: column[i] for name, column in zip(features._fields, columns, strict=True)}}
            for i, (start, _) in enumerate(pending)]


def analyse(out: str | Path, tiou: float = 0.5, min_score: float = 0.3,
            permutations: int = 2000) -> dict:
    """The head's loss per labelled clip and pruned variant against the signals, and the
    annotated actions each variant loses against the dense run."""
    out = Path(out)
    record = RunRecord.read(out)
    rows = _rows(out)
    cut = {(c["video"], c["start"]): c for c in
           (json.loads(line) for line in (out / FOLDER / CUT).read_text("utf-8").splitlines())}
    database = json.loads((out / "ama" / "labels.json").read_text("utf-8"))["database"]
    actions = {stem: [(float(a["segment"][0]), float(a["segment"][1]), int(a["label_id"]))
                      for a in database[stem]["annotations"]] for stem in record.stems}
    detections = _detections(out, record.variants)
    probabilities = {v: {stem: np.load(out / "features" / v / f"{stem}.npz")["a_prob"]
                         for stem in record.stems} for v in record.variants}
    keys = [(stem, start) for stem in record.stems for start in sorted(
        r["start_frame"] for r in rows if r["video"] == stem and r["variant"] == "denso")]
    g = _g(np.array([cut[k]["motion"] for k in keys]),
           np.array([cut[k]["redundancy"] for k in keys]))
    by_key = dict(zip(keys, g, strict=True))

    table, report = [], {"tiou": tiou, "min_score": min_score, "variants": {}}
    for variant in record.variants[1:]:
        clips = []
        for stem in record.stems:
            mine = sorted((r for r in rows if r["video"] == stem and r["variant"] == variant),
                          key=lambda r: r["start_frame"])
            fps = record.video(stem)["fps"]
            span = record.settings["frame_step"] * settings.GEOMETRY.frames_per_clip / fps
            labels = clip_labels([r["start_frame"] / fps for r in mine], span, actions[stem])
            kept = [i for i, label in enumerate(labels) if label is not None]
            if not kept:
                continue
            losses = functional_losses(probabilities["denso"][stem][kept],
                                       probabilities[variant][stem][kept],
                                       [labels[i] for i in kept])
            for j, i in enumerate(kept):
                key = (stem, mine[i]["start_frame"])
                clips.append({
                    "video": stem, "driver": driver_of(stem), "start": key[1],
                    "start_s": key[1] / fps, "variant": variant, "label": labels[i],
                    **{name: float(value[j]) for name, value in losses.items()},
                    "drift": 1.0 - mine[i]["cosine_to_dense"], "g": float(by_key[key]),
                    **{name: cut[key][name] for name in ("motion", "redundancy", "probe_mean",
                                                         "probe_p90")},
                })
        table += clips
        report["variants"][variant] = _relate(clips, permutations) | {
            "actions": _lost(actions, detections, variant, tiou, min_score)}

    folder = out / FOLDER
    _write_csv(folder / "clips.csv", table)
    _write_csv(folder / "actions.csv", _action_rows(record, actions, detections, table, tiou,
                                                   min_score))
    (folder / "proxy.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _relate(clips: list[dict], permutations: int) -> dict:
    """Each signal against each loss, pooled and per driver; and what the probe adds."""
    if not clips:
        return {"labelled_clips": 0, "signals": {}}
    column = {name: np.array([c[name] for c in clips], dtype=float) for name in clips[0]
              if isinstance(clips[0][name], (int, float)) and not isinstance(clips[0][name], bool)}
    flips = np.array([c["flip"] for c in clips], dtype=float) > 0.5
    videos = [c["video"] for c in clips]
    drivers = sorted({c["driver"] for c in clips})
    found = {"labelled_clips": len(clips), "flips": int(flips.sum()), "signals": {}}
    for signal in SIGNALS:
        values = column[signal]
        found["signals"][signal] = {
            **{f"rho_{loss}": spearman(values, column[loss]) for loss in LOSSES},
            "p_delta_ce_within_videos": permutation_p(values, column["delta_ce"], videos,
                                                      permutations),
            "rho_delta_ce_by_driver": {
                d: spearman(values[_mask(clips, d)], column["delta_ce"][_mask(clips, d)])
                for d in drivers},
            "auc_flip": auc(values, flips),
        }
    found["probe_beyond_g"] = {
        loss: partial_spearman(column["probe_p90"], column[loss], column["g"]) for loss in LOSSES}
    found["probe_beyond_motion_redundancy"] = {
        loss: partial_spearman(column["probe_p90"], column[loss], column["motion"],
                               column["redundancy"]) for loss in LOSSES}
    found["mean_delta_ce"] = float(column["delta_ce"].mean())
    return found


def _lost(actions: dict, detections: dict | None, variant: str, tiou: float,
          min_score: float) -> dict:
    """How many annotated actions the dense run finds, the variant finds, and the variant
    loses of those the dense run found."""
    if detections is None:
        return {}
    dense = sum(_finds(detections["denso"], stem, action, tiou, min_score)
                for stem in actions for action in actions[stem])
    found = sum(_finds(detections[variant], stem, action, tiou, min_score)
                for stem in actions for action in actions[stem])
    lost = sum(_finds(detections["denso"], stem, action, tiou, min_score)
               and not _finds(detections[variant], stem, action, tiou, min_score)
               for stem in actions for action in actions[stem])
    return {"annotated": sum(len(a) for a in actions.values()), "found_dense": int(dense),
            "found": int(found), "lost": int(lost)}


def _action_rows(record: RunRecord, actions: dict, detections: dict | None, table: list[dict],
                 tiou: float, min_score: float) -> list[dict]:
    """One row per annotated action and pruned variant: kept or lost by the AMA, the best
    overlap and its confidence, and the mean loss and signals of the clips inside it."""
    found = []
    for variant in record.variants[1:]:
        for stem in record.stems:
            for first, last, label in actions[stem]:
                inside = [c for c in table if c["variant"] == variant and c["video"] == stem
                          and c["label"] == label and first <= c["start_s"] < last]
                row = {"variant": variant, "video": stem, "label": label, "start": first,
                       "end": last, "clips": len(inside)}
                if detections is not None:
                    overlap, confidence = _match(detections[variant], stem, (first, last, label),
                                                 min_score)
                    dense, _ = _match(detections["denso"], stem, (first, last, label), min_score)
                    row |= {"found_dense": dense >= tiou, "found": overlap >= tiou,
                            "best_tiou": overlap, "confidence": confidence}
                for name in ("delta_ce", "delta_p", "drift", "g", "motion", "redundancy",
                             "probe_p90"):
                    row[f"mean_{name}"] = (float(np.mean([c[name] for c in inside]))
                                           if inside else None)
                found.append(row)
    return found


def _match(found, stem: str, action: tuple[float, float, int], min_score: float):
    """The best tIoU of the same-class detections of a video with an action, and the
    confidence of that detection."""
    first, last, label = action
    best, confidence = 0.0, None
    for start, end, kind, score in found.get(stem, []):
        if kind != label or score < min_score:
            continue
        union = max(end, last) - min(start, first)
        overlap = max(0.0, min(end, last) - max(start, first)) / union if union > 0 else 0.0
        if overlap > best:
            best, confidence = overlap, score
    return best, confidence


def _finds(found, stem, action, tiou, min_score) -> bool:
    return _match(found, stem, action, min_score)[0] >= tiou


def _detections(out: Path, variants: Sequence[str]) -> dict | None:
    """``{variant: {video: [(start, end, label, score)]}}``, when the AMA has run."""
    paths = {v: out / "ama" / v / "predictions.csv" for v in variants}
    if not all(path.exists() for path in paths.values()):
        return None
    found = {}
    for variant, path in paths.items():
        found[variant] = {}
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                found[variant].setdefault(row["video-id"], []).append(
                    (float(row["t-start"]), float(row["t-end"]), int(float(row["label"])),
                     float(row["score"])))
    return found


def _g(motion: np.ndarray, redundancy: np.ndarray) -> np.ndarray:
    """The legacy score, ranked within the run: more motion and less redundancy rank higher."""
    def rank(values: np.ndarray) -> np.ndarray:
        order = values.argsort(kind="mergesort")
        ranks = np.empty(len(values))
        ranks[order] = np.arange(len(values))
        return ranks / max(len(values) - 1, 1)
    return 0.5 * rank(motion) + 0.5 * (1.0 - rank(redundancy))


def _mask(clips: list[dict], driver: str) -> np.ndarray:
    return np.array([c["driver"] == driver for c in clips])


def _rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "clips.jsonl").read_text("utf-8").splitlines()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
