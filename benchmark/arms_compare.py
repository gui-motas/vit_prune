"""Run: PYTHONPATH=src python benchmark/arms_compare.py --run NAME=FOLDER --out comparison.json"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from vitprune.pipeline.benchmark import report
from vitprune.pipeline.benchmark.settings import RunRecord


def action_table(gt: pd.DataFrame, detections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    gt = gt.reset_index(drop=True)
    matches = {}
    for arm, found in detections.items():
        found = found.reset_index(drop=True)
        eligible = found[found["score"] >= report.MIN_SCORE].reset_index(drop=True)
        best, where = report._best(gt, eligible)
        any_best, any_where = report._best(gt, found)
        matches[arm] = eligible, best, where, found, any_best, any_where
    rows = []
    for action_id, action in gt.iterrows():
        for arm, (eligible, best, where, found, any_best, any_where) in matches.items():
            overlap = best[action_id]
            picked = eligible.iloc[where[action_id]] if overlap > 0 else None
            any_picked = found.iloc[any_where[action_id]] if any_where[action_id] >= 0 else None
            rows.append({"action": action_id, "arm": arm, **action.to_dict(),
                         "recovered": bool(overlap >= report.TIOU),
                         "score": None if picked is None else float(picked["score"]),
                         "tiou": float(overlap),
                         "best_any": None if any_picked is None else {
                             "score": float(any_picked["score"]),
                             "tiou": float(any_best[action_id])},
                         "start_err": None if picked is None else
                         float(picked["t-start"] - action["t-start"]),
                         "end_err": None if picked is None else
                         float(picked["t-end"] - action["t-end"])})
    return pd.DataFrame(rows)


def false_positive_table(gt: pd.DataFrame, detections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for arm, found in detections.items():
        kept = found[found["score"] >= report.MIN_SCORE]
        false = kept[report.best_tiou(kept, gt) < report.TIOU]
        rows.append(false.assign(arm=arm))
    return pd.concat(rows, ignore_index=True)


def latency(frame: pd.DataFrame, batch: int) -> dict:
    first_video = frame.iloc[0]["video"]
    result = {}
    for arm, rows in frame.groupby("variant", sort=False):
        warmup = rows[rows["video"] == first_video].head(batch).index
        seconds = rows.drop(warmup)["seconds"]
        if seconds.empty:
            raise ValueError(f"{arm}: no timing rows after the first batch")
        result[arm] = {"p50": float(seconds.median()), "p95": float(seconds.quantile(.95)),
                       "mean": float(seconds.mean()), "clips": len(seconds)}
    return result


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def compatibility(runs: dict[str, dict]) -> dict:
    fields = ("weights.sha256", "region.sha256", "schedules", "settings.stride",
              "settings.frame_step", "settings.batch", "settings.prune_layer", "settings.evaluate")
    values, mismatches = {}, []
    for name, run in runs.items():
        settings = run.get("settings") or {}
        values[name] = {"weights.sha256": (run.get("weights") or {}).get("sha256"),
                        "region.sha256": (run.get("region") or {}).get("sha256"),
                        "schedules": run.get("schedules"),
                        **{f"settings.{key}": settings.get(key) for key in
                           ("stride", "frame_step", "batch", "prune_layer", "evaluate")},
                        "attention": settings.get("attention", "model")}
        for field in fields:
            if values[name][field] is None:
                fallback = " (latency assumes 8)" if field == "settings.batch" else ""
                mismatches.append(f"{name}: missing {field}{fallback}")
        for stem, schedule in (run.get("schedules") or {}).items():
            for key in ("clips", "frames", "fps"):
                if schedule.get(key) is None:
                    mismatches.append(f"{name}: missing schedules.{stem}.{key}")
    if values:
        baseline = next(iter(values))
        for name in list(values)[1:]:
            for field in fields:
                left, right = values[baseline][field], values[name][field]
                if left is not None and right is not None and left != right:
                    mismatches.append(f"{field}: {baseline}={left!r}, {name}={right!r}")
    return {"compatible": not mismatches, "mismatches": mismatches, "runs": values}


def markdown(result: dict) -> str:
    arms, actions = result["arms"], result["actions"]
    lines = ["# Arm comparison", ""]
    mismatches = result.get("compatibility", {}).get("mismatches", [])
    if mismatches:
        lines += ["**WARNING: incompatible or missing run provenance**", "",
                  *(f"- {item}" for item in mismatches), ""]
    lines += ["Latency in seconds per clip; first batch excluded.", "",
             "| Arm | Recovered | Lost | FP | mAP | p50 | p95 | Mean |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in arms.items():
        t = row["latency"]
        lines.append(f"| {name} | {row['recovered']} | {row['lost']} | {row['false_positives']} "
                     f"| {row['map']:.4f} | {t['p50']:.4f} | {t['p95']:.4f} "
                     f"| {t['mean']:.4f} |")
    lines += ["", "## Actions", "", "Signed start/end errors are prediction minus annotation. "
              "Lost actions show best same-class score @tIoU, regardless of score threshold.", "",
              "| Video | Class | Start | End | " + " | ".join(arms) + " |",
              "|---|---:|---:|---:|" + "---|" * len(arms)]
    for action in actions:
        cells = []
        for name in arms:
            row = action["arms"][name]
            mark = "✓" if row["recovered"] else "✗"
            if not row["recovered"]:
                best = row.get("best_any")
                cells.append(f"✗ ({best['score']:.2f} @{best['tiou']:.2f})" if best else "✗ —")
            else:
                cells.append(f"{mark} {row['score']:.2f} ({row['tiou']:.2f}) "
                             f"{row['start_err']:+.1f}/{row['end_err']:+.1f}s")
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
    found, scores, timings, truth, runs = {}, {}, {}, None, {}
    run_names = set()
    keys = ["video-id", "t-start", "t-end", "label", "score"]
    for arg in args.run:
        name, sep, folder = arg.partition("=")
        if not sep or not name or not folder or name in run_names:
            parser.error(f"invalid or duplicate run: {arg}")
        run_names.add(name)
        record = RunRecord.read(folder)
        runs[name] = record.run
        gt, detections, maps = (report.ground_truth(record), report.ama_detections(record),
                                report.ama_scores(record))
        if gt is None or detections is None or maps is None:
            raise SystemExit(f"{folder}: missing AMA labels, detections or mAP")
        gt = gt.sort_values(keys).reset_index(drop=True)
        if truth is not None and not truth.equals(gt):
            raise SystemExit(f"{folder}: annotations differ from the other runs")
        truth = gt
        times = latency(report.load(record), record.settings.get("batch", 8))
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
                       ("recovered", "score", "tiou", "start_err", "end_err", "best_any")}
                       for row in records(status)}})
    provenance = compatibility(runs)
    if provenance["mismatches"]:
        print("WARNING: incompatible or missing run provenance:\n- "
              + "\n- ".join(provenance["mismatches"]), flush=True)
    result = {"arms": arms, "actions": actions, "compatibility": provenance}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
