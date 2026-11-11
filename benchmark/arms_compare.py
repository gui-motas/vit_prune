"""Run: PYTHONPATH=src python benchmark/arms_compare.py --run NAME=FOLDER --out comparison.json"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vitprune.pipeline.benchmark import report
from vitprune.pipeline.benchmark.settings import RunRecord


def action_table(gt: pd.DataFrame, detections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for action_id, action in gt.reset_index(drop=True).iterrows():
        for arm, found in detections.items():
            same = found[(found["video-id"] == action["video-id"])
                         & (found["label"] == action["label"])].copy()
            inter = (np.minimum(same["t-end"], action["t-end"])
                     - np.maximum(same["t-start"], action["t-start"])).clip(lower=0)
            union = (np.maximum(same["t-end"], action["t-end"])
                     - np.minimum(same["t-start"], action["t-start"]))
            same["tiou"] = np.divide(inter, union, out=np.zeros(len(inter)), where=union > 0)
            qualified = same[same["tiou"] >= report.TIOU]
            fallback = qualified.empty
            candidates = same[same["tiou"] > 0] if fallback else qualified
            picked = None if candidates.empty else candidates.loc[candidates["score"].idxmax()]
            rows.append({"action": action_id, "arm": arm, **action.to_dict(),
                         "recovered": bool(picked is not None and not fallback
                                           and picked["score"] >= report.MIN_SCORE),
                         "score": None if picked is None else float(picked["score"]),
                         "tiou": None if picked is None else float(picked["tiou"]),
                         "start_err": None if picked is None else
                         float(picked["t-start"] - action["t-start"]),
                         "end_err": None if picked is None else
                         float(picked["t-end"] - action["t-end"]),
                         "fallback": bool(fallback and picked is not None)})
    return pd.DataFrame(rows)


def false_positive_table(gt: pd.DataFrame, detections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for arm, found in detections.items():
        kept = found[found["score"] >= report.MIN_SCORE]
        false = kept[report.best_tiou(kept, gt) < report.TIOU]
        rows.append(false.assign(arm=arm))
    return pd.concat(rows, ignore_index=True)


def latency(frame: pd.DataFrame) -> dict:
    first_video = frame.iloc[0]["video"]
    result = {}
    for arm, rows in frame.groupby("variant", sort=False):
        warmup = rows[rows["video"] == first_video].head(8).index
        seconds = rows.drop(warmup)["seconds"]
        if seconds.empty:
            raise ValueError(f"{arm}: no timing rows after the first batch")
        result[arm] = {"p50": float(seconds.median()), "p95": float(seconds.quantile(.95)),
                       "mean": float(seconds.mean()), "clips": len(seconds)}
    return result


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def markdown(result: dict) -> str:
    arms, actions = result["arms"], result["actions"]
    lines = ["# Arm comparison", "", "Latency in seconds per clip; first batch excluded.", "",
             "| Arm | Recovered | Lost | FP | mAP | p50 | p95 | Mean |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in arms.items():
        t = row["latency"]
        lines.append(f"| {name} | {row['recovered']} | {row['lost']} | {row['false_positives']} "
                     f"| {row['map']:.4f} | {t['p50']:.4f} | {t['p95']:.4f} "
                     f"| {t['mean']:.4f} |")
    lines += ["", "## Actions", "", "* marks a same-class overlap below the tIoU threshold.", "",
              "| Video | Class | Start | End | " + " | ".join(arms) + " |",
              "|---|---:|---:|---:|" + "---|" * len(arms)]
    for action in actions:
        cells = []
        for name in arms:
            row = action["arms"][name]
            mark = "✓" if row["recovered"] else "✗"
            cells.append(f"{mark} {row['score']:.2f} ({row['tiou']:.2f})"
                         + ("*" if row["fallback"] else "")
                         if row["score"] is not None else f"{mark} —")
        lines.append(f"| {action['video-id']} | {action['label']} | {action['t-start']:.2f} "
                     f"| {action['t-end']:.2f} | " + " | ".join(cells) + " |")
    lines += ["", "## False positives", ""]
    for name, row in arms.items():
        lines += [f"### {name}", "", "| Video | Class | Start | End | Score |",
                  "|---|---:|---:|---:|---:|"]
        for fp in row["detections"]:
            lines.append(f"| {fp['video-id']} | {fp['label']} | {fp['t-start']:.2f} "
                         f"| {fp['t-end']:.2f} | {fp['score']:.2f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=FOLDER")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    found, scores, timings, truth = {}, {}, {}, None
    run_names = set()
    keys = ["video-id", "t-start", "t-end", "label", "score"]
    for arg in args.run:
        name, sep, folder = arg.partition("=")
        if not sep or not name or not folder or name in run_names:
            parser.error(f"invalid or duplicate run: {arg}")
        run_names.add(name)
        record = RunRecord.read(folder)
        gt, detections, maps = (report.ground_truth(record), report.ama_detections(record),
                                report.ama_scores(record))
        if gt is None or detections is None or maps is None:
            raise SystemExit(f"{folder}: missing AMA labels, detections or mAP")
        gt = gt.sort_values(keys).reset_index(drop=True)
        if truth is not None and not truth.equals(gt):
            raise SystemExit(f"{folder}: annotations differ from the other runs")
        truth = gt
        times = latency(report.load(record))
        for variant in record.variants:
            arm = f"{name}/{variant}"
            found[arm], scores[arm] = detections[variant], maps[variant]
            timings[arm] = times[variant]
        print(f"read {name}: {len(gt)} actions, {len(record.variants)} arms", flush=True)
    table, fps = action_table(truth, found), false_positive_table(truth, found)
    arms = {}
    for arm in found:
        status = table[table["arm"] == arm]
        false = fps[fps["arm"] == arm].drop(columns="arm")
        arms[arm] = {"recovered": int(status["recovered"].sum()),
                     "lost": int((~status["recovered"]).sum()), "false_positives": len(false),
                     "map": scores[arm]["average"], "latency": timings[arm],
                     "detections": records(false)}
    actions = []
    for action_id, action in truth.iterrows():
        status = table[table["action"] == action_id]
        actions.append(action.to_dict() | {"arms": {row["arm"]: {k: row[k] for k in
                       ("recovered", "score", "tiou", "start_err", "end_err", "fallback")}
                       for row in records(status)}})
    result = {"arms": arms, "actions": actions}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
