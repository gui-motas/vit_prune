"""Video-level check: does the AMA's final answer survive ViT-Prune with the driver region?

Takes one stretch of a video holding two actions in a row, runs it dense and with
ViT-Prune plus the driver region, and hands each sequence of clip embeddings to the AMA
temporal localizer, as the AI City pipeline does. What is compared is the AMA's output:
which actions it finds, where, and with what confidence.

The policy and the region are the ones in the settings, as in ``run_inference.py``. The
AMA runs on CPU through ``nms_1d_cpu.py`` when its compiled extension does not load.
Writes ``results/video-level/``.
"""

from __future__ import annotations

import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from common import (  # noqa: E402
    FRAME_STEP,
    GEOMETRY,
    RESULTS,
    SETTINGS,
    SPAN,
    STRIDE,
    segments,
    videos,
)
from run_inference import POLICY, PRUNE_LAYER, REGION, load_model  # noqa: E402

from vitprune.core import PrunedEncoder, PruningPlan  # noqa: E402
from vitprune.pipeline.calibration import read_policy  # noqa: E402
from vitprune.pipeline.region import read_region  # noqa: E402
from vitprune.pipeline.video import iter_clips  # noqa: E402

VIDEO = "Dashboard_user_id_13522_NoAudio_5"
ACTIONS = ("Drinking", "yawning")
MARGIN = 6.0  # seconds of context before the first action and after the last
OUT = RESULTS / "video-level"
SHOWN_SCORE = 0.3


def load_ama():
    sys.path.insert(0, SETTINGS["ama_project"])
    from libs.core import load_config
    from libs.modeling import make_meta_arch

    config = load_config(SETTINGS["ama_config"])
    model = make_meta_arch(config["model_name"], **config["model"])
    raw = torch.load(SETTINGS["ama_weights"], map_location="cpu", weights_only=False)
    state = raw.get("state_dict_ema") or raw["state_dict"]
    missing, _ = model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state.items()}, strict=False
    )
    if missing:
        raise ValueError(f"AMA weights incomplete: {missing[:5]}")
    return model.eval(), config


@torch.inference_mode()
def localize(ama, config, feats: np.ndarray, duration: float, names: dict) -> list[dict]:
    dataset = config["dataset"]
    sample = {
        "video_id": VIDEO, "feats": torch.from_numpy(feats.T).float(), "segments": None,
        "labels": None, "segmentation_labels": torch.zeros((feats.shape[0], len(names))),
        "fps": float(dataset["default_fps"]), "duration": duration,
        "feat_stride": dataset["feat_stride"], "feat_num_frames": dataset["num_frames"],
    }
    found = ama([sample], is_training=False)[0]
    rows = [
        {"label_id": int(label), "label": names[int(label)], "start": float(span[0]),
         "end": float(span[1]), "score": float(score)}
        for span, score, label in zip(found["segments"].tolist(), found["scores"].tolist(),
                                      found["labels"].tolist(), strict=True)
    ]
    return sorted(rows, key=lambda r: -r["score"])


def tiou(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return inter / (max(a["end"], b["end"]) - min(a["start"], b["start"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    video = next(v for v in videos() if v.stem == VIDEO)
    annotated = segments(VIDEO)
    names = {s["label_id"]: s["label"] for s in annotated}
    chosen = [s for s in annotated if s["label"] in ACTIONS]
    first, last = max(0.0, chosen[0]["start"] - MARGIN), chosen[-1]["end"] + MARGIN
    fps = 30.0
    # Clips start on the stride grid, as the extraction for the AMA samples them.
    begin, end = -(-round(first * fps) // STRIDE) * STRIDE, round(last * fps)
    starts = list(range(begin, end - SPAN + 1, STRIDE))
    truth = [{"label": s["label"], "start": s["start"] - first, "end": s["end"] - first}
             for s in annotated if first <= s["start"] and s["end"] <= last]
    print(f"{VIDEO} {first:.0f}-{last:.0f}s: {len(starts)} clips, "
          f"actions {[t['label'] for t in truth]}")

    model = load_model()
    policy = read_policy(POLICY, PRUNE_LAYER, FRAME_STEP)
    region = read_region(REGION, GEOMETRY)
    runners = {
        "denso": PrunedEncoder(model, GEOMETRY, PruningPlan.dense()),
        "vitprune_regiao": PrunedEncoder(model, GEOMETRY, PruningPlan(
            prune_layer=PRUNE_LAYER, budget=policy, spatial_prior=region)),
    }
    feats = {name: [] for name in runners}
    clips, seconds = [], {name: 0.0 for name in runners}
    with torch.inference_mode():
        for position, (start, clip) in enumerate(
            iter_clips(video, GEOMETRY, STRIDE, FRAME_STEP, starts), start=1
        ):
            row = {"start_s": round(start / fps - first, 2)}
            dense = None
            for name, encoder in runners.items():
                started = time.perf_counter()
                if encoder.plan.prunes:
                    embedding, budget = encoder.encode_with_budget(clip[None])
                    row["tokens"] = int(budget.tokens[0])
                else:
                    embedding = encoder.encode(clip[None])
                seconds[name] += time.perf_counter() - started
                dense = embedding if dense is None else dense
                feats[name].append(embedding[0].numpy())
                row[f"{name}_clipe"] = names[int(model.head(embedding).argmax())]
            row["cosseno"] = float(F.cosine_similarity(embedding, dense).item())
            clips.append(row)
            print(f"  [{position}/{len(starts)}] {row['start_s']:.1f}s cos {row['cosseno']:.3f} "
                  f"denso={row['denso_clipe']} regiao={row['vitprune_regiao_clipe']}", flush=True)

    ama, config = load_ama()
    duration = last - first
    found = {}
    for name in runners:
        stacked = np.stack(feats[name])
        np.savez(OUT / f"embeddings-{name}.npz", feats=stacked)
        found[name] = localize(ama, config, stacked, duration, names)

    matches = []
    for action in truth:
        row = {"ação": action["label"], "início real (s)": action["start"] + first,
               "fim real (s)": action["end"] + first}
        for name, detections in found.items():
            same = [d for d in detections if d["label"] == action["label"]]
            best = max(same, key=lambda d: tiou(d, action), default=None)
            row[f"{name}: início"] = None if best is None else round(best["start"] + first, 2)
            row[f"{name}: fim"] = None if best is None else round(best["end"] + first, 2)
            row[f"{name}: confiança"] = None if best is None else round(best["score"], 3)
            row[f"{name}: tIoU"] = 0.0 if best is None else round(tiou(best, action), 3)
        matches.append(row)

    top = {name: [d for d in detections if d["score"] >= SHOWN_SCORE]
           for name, detections in found.items()}
    summary = {
        "video": VIDEO, "window_s": [first, last], "clips": len(starts),
        "truth": [{**t, "start": t["start"] + first, "end": t["end"] + first} for t in truth],
        "policy": policy.version, "region_area": float(region.mean()),
        "tokens": sorted({c["tokens"] for c in clips}),
        "cosine_mean": float(np.mean([c["cosseno"] for c in clips])),
        "seconds": seconds, "speedup": seconds["denso"] / seconds["vitprune_regiao"],
        "matches": matches,
        "detections": {name: [{**d, "start": d["start"] + first, "end": d["end"] + first}
                              for d in rows] for name, rows in top.items()},
    }
    (OUT / "resumo.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    with pd.ExcelWriter(OUT / "video-inteiro.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(matches).to_excel(writer, sheet_name="Ações reais", index=False)
        for name, rows in summary["detections"].items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=f"AMA {name}"[:31], index=False)
        pd.DataFrame(clips).to_excel(writer, sheet_name="Por clipe", index=False)
    timeline(summary)
    print(json.dumps({k: summary[k] for k in ("matches", "speedup", "cosine_mean", "tokens")},
                     indent=2, ensure_ascii=False))


def timeline(summary: dict) -> None:
    labels = sorted({d["label"] for rows in summary["detections"].values() for d in rows}
                    | {t["label"] for t in summary["truth"]})
    colors = dict(zip(labels, plt.cm.tab10.colors, strict=False))
    lanes = [("Anotação real", [dict(t, score=1.0) for t in summary["truth"]]),
             ("AMA sem ViT-Prune", summary["detections"]["denso"]),
             ("AMA com ViT-Prune + região", summary["detections"]["vitprune_regiao"])]
    figure, axis = plt.subplots(figsize=(14, 3.6))
    for lane, (_, rows) in enumerate(lanes):
        for row in sorted(rows, key=lambda r: r["score"]):
            axis.barh(lane, row["end"] - row["start"], left=row["start"], height=0.6,
                      color=colors[row["label"]], alpha=0.35 + 0.65 * row["score"])
            axis.text((row["start"] + row["end"]) / 2, lane, f"{row['label']}\n{row['score']:.2f}",
                      ha="center", va="center", fontsize=7)
    axis.set_yticks(range(len(lanes)), [title for title, _ in lanes])
    axis.invert_yaxis()
    axis.set_xlabel("segundos no vídeo")
    axis.set_title(f"{summary['video']}: ações encontradas pelo AMA (confiança ≥ {SHOWN_SCORE})")
    figure.tight_layout()
    figure.savefig(OUT / "linha-do-tempo.png", dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    main()
