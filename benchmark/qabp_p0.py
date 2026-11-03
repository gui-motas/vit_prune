"""P0 of the quality-aware policy: before any policy, is the pruned path sound, and how does
fidelity fall with the budget?

Run on calibration clips only, after ``vitprune budget-curve`` wrote their records (see
``run_qabp.sh p0``); nothing here tunes a parameter. Writes, under ``--out``:

* ``p0.json`` -- the backbone check, the identity of the full level with the dense path,
  the latency of each level, the summary of the records and the verdict of gates G1 and G2;
* ``p0-clips.csv`` -- one row per clip and level: the drift, the merge error, the drift of
  the token mean after each recorded block, and the clip's features;
* ``p0-drift.png`` -- the distribution of the cosine to the dense embedding per level;
* ``p0-stages.png`` -- how the drift grows block after block, per level.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from vitprune.bench.latency import autocast, measure_plans  # noqa: E402
from vitprune.core import (  # noqa: E402
    ClipGeometry,
    FixedBudget,
    PrunedEncoder,
    PruningPlan,
    check_backbone,
)
from vitprune.pipeline.checkpoint import (  # noqa: E402
    disable_activation_checkpointing,
    load_videomae,
)
from vitprune.pipeline.quality import p0_verdict, read_records, summarise_records  # noqa: E402
from vitprune.pipeline.video import gather_clips  # noqa: E402

LEVELS = (512, 1024, 1536, 2048)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--records", required=True, help="written by vitprune budget-curve")
    parser.add_argument("--videos", required=True, help="calibration videos for the timed batch")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    settings, _, records = read_records(args.records)
    summary = summarise_records(records, settings["levels"])
    geometry = ClipGeometry.videomae_giant()
    model = disable_activation_checkpointing(
        load_videomae(args.weights, args.model_source, geometry, device=args.device))
    check = check_backbone(model, geometry, reference=model.forward_features)
    print(check.describe(), flush=True)

    clips = gather_clips(args.videos, geometry, settings["stride"], settings["frame_step"], 8)
    with torch.no_grad(), autocast(args.device, "off"):
        batch = clips.to(args.device)
        dense = PrunedEncoder(model, geometry, PruningPlan.dense()).encode(batch)
        full = PrunedEncoder(model, geometry, PruningPlan(
            prune_layer=settings["prune_layer"], budget=FixedBudget(geometry.num_tokens),
        )).encode(batch)
    difference = float((full - dense).abs().max())
    print(f"full level against the dense path: max difference {difference:.3e}", flush=True)

    plans = [PruningPlan.dense()] + [
        PruningPlan(prune_layer=settings["prune_layer"], budget=FixedBudget(level))
        for level in LEVELS
    ]
    latency = measure_plans(model, clips, plans, geometry, warmup=2, iterations=5,
                            device=args.device, precision="off")
    verdict = p0_verdict(summary, difference, check.ok)

    (out / "p0.json").write_text(json.dumps({
        "check": check.describe(), "identity_max_diff": difference,
        "latency": [point.to_dict() for point in latency],
        "summary": summary, "verdict": verdict, "records": str(Path(args.records).resolve()),
    }, indent=2), encoding="utf-8")
    write_clips(out / "p0-clips.csv", records)
    drift_figure(out / "p0-drift.png", records, summary)
    stages_figure(out / "p0-stages.png", summary)
    print(json.dumps(verdict, indent=2))
    return 0


def write_clips(path: Path, records: list[dict]) -> None:
    depths = sorted({depth for record in records for level in record["levels"].values()
                     for depth in level["stages"]}, key=int)
    columns = ["video", "driver", "start", "level", "cos", "d_rel", "top1", "norm_ratio",
               "merge_error_mean", "merge_error_p90", "merge_error_max",
               *[f"cos_after_{depth}" for depth in depths],
               *[f"rel_l2_after_{depth}" for depth in depths],
               "motion", "redundancy", "probe_mean", "probe_p90"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for record in records:
            for level, found in record["levels"].items():
                writer.writerow([
                    record["video"], record["driver"], record["start"], level,
                    1 - found["d_cos"], found["d_rel"], found["top1"],
                    found["norm"] / record["dense"]["norm"],
                    *(found["merge_error"][name] for name in ("mean", "p90", "max")),
                    *(found["stages"][depth]["cos"] for depth in depths),
                    *(found["stages"][depth]["rel_l2"] for depth in depths),
                    record["motion"], record["redundancy"], record["probe_mean"],
                    record["probe_p90"],
                ])


def drift_figure(path: Path, records: list[dict], summary: dict) -> None:
    levels = list(summary["levels"])
    values = [[1 - record["levels"][level]["d_cos"] for record in records] for level in levels]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.boxplot(values, whis=(10, 90), showfliers=True)
    for x, level in enumerate(levels, start=1):
        cos = summary["levels"][level]["cos"]
        axis.text(x, cos["p10"], f"P10 {cos['p10']:.2f}\nP50 {cos['p50']:.2f}", fontsize=8,
                  ha="center", va="top")
    axis.set_xticks(range(1, len(levels) + 1), [f"{level} tokens" for level in levels])
    axis.axhline(0.90, color="red", linewidth=0.8, linestyle="--")
    axis.set_ylabel("cosseno com o embedding denso")
    axis.set_title(f"P0: {summary['clips']} clipes de calibração (caixa P25–P75, hastes P10–P90)")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def stages_figure(path: Path, summary: dict) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    for level, found in summary["levels"].items():
        depths = sorted(found["stages"], key=int)
        left.plot([int(d) for d in depths], [found["stages"][d]["cos_p50"] for d in depths],
                  marker="o", label=f"{level} tokens")
        right.plot([int(d) for d in depths], [found["stages"][d]["rel_l2_p50"] for d in depths],
                   marker="o", label=f"{level} tokens")
    left.set_title("cosseno da média dos tokens com o denso (mediana)")
    right.set_title("L2 relativo da média dos tokens (mediana)")
    for axis in (left, right):
        axis.set_xlabel("depois do bloco")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
