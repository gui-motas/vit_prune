"""Turn the benchmark records into a spreadsheet and figures.

Reads what ``vitprune benchmark run`` and, once it has run, ``vitprune benchmark ama``
wrote under the results folder, and writes there:

* ``benchmark-vitprune.xlsx`` -- the summary per variant; every detection of the dense
  pipeline read whole, per variant; the tokens kept per video; and the configuration the
  numbers were produced under;
* ``figuras/`` -- the fixed and the smart mask over a frame of each video, the tokens each
  variant kept in clips spread over each video, the AMA's detections along each video and a
  chart of the summary.

The dense pipeline is a reference, in two places: the class the classification head gives
each clip, and the detections of the AMA. A variant "agrees" when it gives what the dense
gave; that measures how much pruning changes the result, not whether the result is right.
When ``vitprune benchmark ama`` was given the annotations, each variant is also judged
against them: the AMA's mAP, the annotated actions it finds, the error of their boundaries,
its false positives, and the class the head gives each annotated clip and action.

Three things are kept apart and never read as one another: how far the embedding moved
from the dense one, how much the answers agree with the dense ones, and how well the
actions are found.

Needs the ``benchmark`` extra: pandas, openpyxl and matplotlib.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from vitprune.bench.cost import budget_cost  # noqa: E402
from vitprune.core.policy import REASONS  # noqa: E402
from vitprune.pipeline.benchmark.settings import GEOMETRY, VARIANTS, RunRecord  # noqa: E402

NAMES = {
    "denso": "Sem ViT-Prune (denso, referência)",
    "vitprune": "ViT-Prune sem máscara",
    "vitprune_mascara_fixa": "ViT-Prune + máscara fixa",
    "vitprune_mascara_inteligente": "ViT-Prune + máscara inteligente",
    "legacy_original": "ViT-Prune legacy original (policy.json)",
    "legacy_recalibrada": "ViT-Prune legacy recalibrada no ajuste",
    "qabp": "ViT-Prune QABP",
    "qabp_sem_probe": "ViT-Prune QABP sem probe (ablação)",
}
SHORT = {
    "denso": "Denso",
    "vitprune": "Sem máscara",
    "vitprune_mascara_fixa": "Máscara fixa",
    "vitprune_mascara_inteligente": "Máscara inteligente",
    "legacy_original": "Legacy original",
    "legacy_recalibrada": "Legacy recalibrada",
    "qabp": "QABP",
    "qabp_sem_probe": "QABP sem probe",
}


def _fixed(variant: str) -> tuple[int, str] | None:
    """The tokens of a fixed-budget variant and what it adds: ``pa`` (size-biased attention),
    ``mascara`` (the driver region) or nothing."""
    found = re.fullmatch(r"fixo_(\d+)(?:_(pa|mascara))?", variant)
    return (int(found.group(1)), found.group(2) or "") if found else None


def variant_name(variant: str) -> str:
    """A variant's name in the tables."""
    fixed = _fixed(variant)
    if fixed:
        extra = {"pa": " + atenção proporcional", "mascara": " + máscara do motorista"}
        return f"ViT-Prune fixo {fixed[0]} tokens" + extra.get(fixed[1], "")
    return NAMES.get(variant, variant)


def short_label(variant: str) -> str:
    """A variant's name for a chart axis; the dense one says it is the reference."""
    if variant == "denso":
        return "Denso (referência)"
    fixed = _fixed(variant)
    if fixed:
        return f"Fixo {fixed[0]}" + {"pa": " + AP", "mascara": " + máscara"}.get(fixed[1], "")
    return SHORT.get(variant, variant)


COLORS = {"denso": "#7f7f7f", "vitprune": "#1f77b4", "vitprune_mascara_fixa": "#ff7f0e",
          "vitprune_mascara_inteligente": "#d62728"}


def color(variant: str, index: int):
    return COLORS.get(variant, plt.cm.tab10.colors[index % 10])


PRUNED = VARIANTS[1:]
MASKS = "vitprune_mascara_inteligente"
GRID = GEOMETRY.grid_size
MIDDLE_SLICE = GEOMETRY.temporal_slices // 2
#: Detections below this confidence are left out, in the reference and in each variant.
MIN_SCORE = 0.3
#: A detection of a variant matches one of the dense when the same class overlaps this much.
TIOU = 0.5
CLASSES = "Concordância de classe com o denso"
RECALL = "Detecções do denso reencontradas (recall)"
PRECISION = "Detecções confirmadas pelo denso (precisão)"
MEAN_TIOU = "tIoU médio com o denso"
GT_RECALL = "Ações anotadas reencontradas (recall)"
GT_PRECISION = "Detecções confirmadas pelas anotações (precisão)"
PA_COST = "Custo extra da atenção proporcional (%)"
FALLBACK = "Clipes no fallback denso (%)"
#: Reasons a quality-aware policy gives for the dense path when it does not trust itself.
FALLBACKS = REASONS[2:]
#: Columns shown as percentages in the workbook.
PERCENT = ("Concordância", "recall", "precisão", "(%)", "região", "Acurácia", "mAP")


def load(record: RunRecord) -> pd.DataFrame:
    rows = [json.loads(line)
            for line in (record.out / "clips.jsonl").read_text("utf-8").splitlines()]
    return pd.DataFrame(rows)


def recorded(record: RunRecord) -> tuple[dict | None, dict]:
    """The policy and the region files the run names; no policy for a policy study without
    one."""
    path = (record.run.get("policy") or {}).get("path")
    policy = json.loads(Path(path).read_text("utf-8")) if path else None
    region = json.loads(Path(record.run["region"]["path"]).read_text("utf-8"))
    return policy, region


def _every(paths: dict[str, Path], what: str) -> bool:
    """Whether every variant has its AMA output; saying so when only some have, which is what
    an AMA step cut short leaves, so a workbook without the task metrics does not look whole."""
    missing = [variant for variant, path in paths.items() if not path.exists()]
    if missing and len(missing) < len(paths):
        print(f"warning: no AMA {what} for {missing}; the report leaves the AMA's {what} out "
              "for every variant", flush=True)
    return not missing


def ama_detections(record: RunRecord) -> dict[str, pd.DataFrame] | None:
    """The AMA's detections per variant, in seconds, when ``vitprune benchmark ama`` has run."""
    paths = {variant: record.out / "ama" / variant / "predictions.csv"
             for variant in record.variants}
    if not _every(paths, "detections"):
        return None
    return {variant: pd.read_csv(path) for variant, path in paths.items()}


def ground_truth(record: RunRecord) -> pd.DataFrame | None:
    """The annotated actions of the evaluated videos, when ``ama --labels`` was given them."""
    path = record.out / "ama" / "labels.json"
    if not path.exists():
        return None
    database = json.loads(path.read_text("utf-8"))["database"]
    return pd.DataFrame(
        [[stem, float(a["segment"][0]), float(a["segment"][1]), int(a["label_id"]), 1.0]
         for stem, video in database.items() for a in video["annotations"]],
        columns=["video-id", "t-start", "t-end", "label", "score"],
    )


def ama_scores(record: RunRecord) -> dict[str, dict] | None:
    """The AMA's mAP against the annotations per variant, when it was computed."""
    paths = {variant: record.out / "ama" / variant / "map.json" for variant in record.variants}
    if not _every(paths, "mAP"):
        return None
    return {variant: json.loads(path.read_text("utf-8")) for variant, path in paths.items()}


def head_probabilities(record: RunRecord, frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """The head's class probabilities per variant and video, one row per clip in clip order."""
    found = {}
    for variant in record.variants:
        found[variant] = {}
        for stem in record.stems:
            probabilities = np.load(record.out / "features" / variant / f"{stem}.npz")["a_prob"]
            clips = ((frame["video"] == stem) & (frame["variant"] == variant)).sum()
            if len(probabilities) != clips:
                raise ValueError(f"{variant}/{stem}.npz holds {len(probabilities)} clips, "
                                 f"clips.jsonl {clips}; the run is incomplete")
            found[variant][stem] = probabilities
    return found


def best_tiou(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """For each detection in ``a``, the highest tIoU with a detection of ``b`` of the same
    video and class; 0 when there is none."""
    return _best(a, b)[0]


def _best(a: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For each row of ``a``, the best tIoU with a row of ``b`` of the same video and class,
    and that row's position in ``b``; ``-1`` when there is none."""
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    best, where = np.zeros(len(a)), np.full(len(a), -1)
    if a.empty or b.empty:
        return best, where
    for (video, label), rows in a.groupby(["video-id", "label"]):
        other = b[(b["video-id"] == video) & (b["label"] == label)]
        if other.empty:
            continue
        start, end = rows["t-start"].to_numpy()[:, None], rows["t-end"].to_numpy()[:, None]
        other_start, other_end = other["t-start"].to_numpy(), other["t-end"].to_numpy()
        inter = np.clip(np.minimum(end, other_end) - np.maximum(start, other_start), 0, None)
        union = np.maximum(end, other_end) - np.minimum(start, other_start)
        overlap = np.divide(inter, union, out=np.zeros(inter.shape), where=union > 0)
        best[rows.index.to_numpy()] = overlap.max(axis=1)
        where[rows.index.to_numpy()] = other.index.to_numpy()[overlap.argmax(axis=1)]
    return best, where


def boundary_errors(reference: pd.DataFrame, found: pd.DataFrame, tiou: float = TIOU,
                    min_score: float = MIN_SCORE) -> dict:
    """How far the boundaries of the detections that find a reference action sit from it:
    the mean absolute error of the start and of the end, in seconds, over the matched ones."""
    reference = reference[reference["score"] >= min_score].reset_index(drop=True)
    found = found[found["score"] >= min_score].reset_index(drop=True)
    best, where = _best(reference, found)
    matched = best >= tiou
    if not matched.any():
        return {"matched": 0, "start": None, "end": None}
    pairs, actions = found.loc[where[matched]], reference[matched]
    return {
        "matched": int(matched.sum()),
        "start": float(np.abs(pairs["t-start"].to_numpy() - actions["t-start"].to_numpy()).mean()),
        "end": float(np.abs(pairs["t-end"].to_numpy() - actions["t-end"].to_numpy()).mean()),
    }


def false_positives(reference: pd.DataFrame, found: pd.DataFrame, tiou: float = TIOU,
                    min_score: float = MIN_SCORE) -> int:
    """Detections from ``min_score`` up that match no reference action of their class."""
    found = found[found["score"] >= min_score]
    reference = reference[reference["score"] >= min_score]
    return int((best_tiou(found, reference) < tiou).sum())


def matched_confidence(reference: pd.DataFrame, found: pd.DataFrame, tiou: float = TIOU,
                       min_score: float = MIN_SCORE) -> float | None:
    """The mean confidence of the detections that find a reference action."""
    reference = reference[reference["score"] >= min_score].reset_index(drop=True)
    found = found[found["score"] >= min_score].reset_index(drop=True)
    best, where = _best(reference, found)
    matched = best >= tiou
    return float(found.loc[where[matched], "score"].mean()) if matched.any() else None


def shared_actions(reference: pd.DataFrame, found: pd.DataFrame, dense: pd.DataFrame,
                   tiou: float = TIOU, min_score: float = MIN_SCORE) -> pd.DataFrame:
    """The reference actions that both ``found`` and ``dense`` find: the only ones on which
    two runs' boundaries and confidence can be set side by side. Over each run's own finds, a
    run that loses the hard actions would look closer."""
    reference = reference[reference["score"] >= min_score].reset_index(drop=True)
    mine = _best(reference, found[found["score"] >= min_score])[0]
    theirs = _best(reference, dense[dense["score"] >= min_score])[0]
    return reference[(mine >= tiou) & (theirs >= tiou)]


def clip_accuracy(probabilities: np.ndarray, centres: np.ndarray,
                  actions: pd.DataFrame) -> tuple[int, int]:
    """Clips whose centre lies in an annotated action and whose top class is that action's,
    and the clips whose centre lies in one."""
    correct = counted = 0
    for found, centre in zip(probabilities, centres, strict=True):
        inside = actions[(actions["t-start"] <= centre) & (centre <= actions["t-end"])]
        if not inside.empty:
            counted += 1
            correct += int(found.argmax()) in set(inside["label"].astype(int))
    return correct, counted


def agreement(reference: pd.DataFrame, found: pd.DataFrame, tiou: float = TIOU,
              min_score: float = MIN_SCORE) -> dict:
    """How the detections ``found`` agree with the ``reference``, both from ``min_score`` up.

    ``recall`` is the share of the reference that ``found`` has again at ``tiou``;
    ``precision``, the share of ``found`` that the reference has; ``mean_tiou``, the best
    tIoU of each reference detection, 0 for the ones not found, averaged. ``None`` when the
    side a figure divides by is empty.
    """
    reference = reference[reference["score"] >= min_score]
    found = found[found["score"] >= min_score]
    toward, back = best_tiou(reference, found), best_tiou(found, reference)
    return {
        "recall": None if reference.empty else float((toward >= tiou).mean()),
        "precision": None if found.empty else float((back >= tiou).mean()),
        "mean_tiou": None if reference.empty else float(toward.mean()),
    }


def class_agreement(dense: np.ndarray, other: np.ndarray) -> float:
    """The share of clips where the head's top class is the dense one's."""
    if len(dense) != len(other):
        raise ValueError(f"{len(other)} clips against the dense run's {len(dense)}")
    return float((dense.argmax(axis=1) == other.argmax(axis=1)).mean())


def window_class(probabilities: np.ndarray, starts_s: np.ndarray, span_s: float,
                 window: tuple[float, float]) -> int | None:
    """The class the head gives a stretch of video: the top class of the mean probabilities
    over the clips lying wholly inside the window; ``None`` when there are none."""
    inside = (starts_s >= window[0]) & (starts_s + span_s <= window[1])
    return int(probabilities[inside].mean(axis=0).argmax()) if inside.any() else None


def _starts(frame: pd.DataFrame, variant: str, stem: str) -> np.ndarray:
    rows = frame[(frame["video"] == stem) & (frame["variant"] == variant)]
    return rows.sort_values("start_frame")["start_s"].to_numpy()


def _span(record: RunRecord, stem: str) -> float:
    return record.settings["frame_step"] * GEOMETRY.frames_per_clip / record.video(stem)["fps"]


def annotated(record: RunRecord, frame: pd.DataFrame, probabilities: dict, variant: str,
              actions: pd.DataFrame, stems: list[str] | None = None) -> dict:
    """The head against the annotations: per clip, by the action holding its centre, and per
    action, by the mean of the clips wholly inside it."""
    correct = counted = right = whole = 0
    for stem in record.stems if stems is None else stems:
        mine = actions[actions["video-id"] == stem]
        starts, span = _starts(frame, variant, stem), _span(record, stem)
        hits, clips = clip_accuracy(probabilities[variant][stem], starts + span / 2, mine)
        correct, counted = correct + hits, counted + clips
        for _, action in mine.iterrows():
            found = window_class(probabilities[variant][stem], starts, span,
                                 (action["t-start"], action["t-end"]))
            if found is not None:
                whole += 1
                right += int(found == int(action["label"]))
    return {"clip": correct / counted if counted else None,
            "action": right / whole if whole else None}


def task_metrics(record: RunRecord, frame: pd.DataFrame, probabilities: dict,
                 detections: dict | None, actions: pd.DataFrame, variant: str, tiou: float,
                 min_score: float, stems: list[str] | None = None) -> dict:
    """A variant against the annotated actions of ``stems`` (every video by default)."""
    stems = record.stems if stems is None else stems
    mine = actions[actions["video-id"].isin(stems)]
    row = {}
    if detections is not None:
        found = detections[variant][detections[variant]["video-id"].isin(stems)]
        dense = detections["denso"][detections["denso"]["video-id"].isin(stems)]
        got = agreement(mine, found, tiou, min_score)
        common = shared_actions(mine, found, dense, tiou, min_score)
        edges = boundary_errors(common, found, tiou, min_score)
        dense_edges = boundary_errors(common, dense, tiou, min_score)
        row |= {GT_RECALL: got["recall"], GT_PRECISION: got["precision"],
                "Falsos positivos": false_positives(mine, found, tiou, min_score),
                "Ações em comum com o denso": len(common),
                "Erro médio no início (s)": edges["start"],
                "Erro médio no fim (s)": edges["end"],
                "Confiança média das reencontradas": matched_confidence(common, found, tiou,
                                                                        min_score),
                "Denso nas mesmas ações: erro no início (s)": dense_edges["start"],
                "Denso nas mesmas ações: erro no fim (s)": dense_edges["end"],
                "Denso nas mesmas ações: confiança": matched_confidence(common, dense, tiou,
                                                                       min_score)}
    head = annotated(record, frame, probabilities, variant, mine, stems)
    return row | {"Acurácia de classe por clipe (anotações)": head["clip"],
                  "Acurácia de classe por ação (anotações)": head["action"]}


def summary(record: RunRecord, frame: pd.DataFrame, probabilities: dict,
            detections: dict | None, area: float, tiou: float, min_score: float,
            actions: pd.DataFrame | None = None, scores: dict | None = None) -> pd.DataFrame:
    dense_seconds = frame[frame["variant"] == "denso"]["seconds"].mean()

    def pooled(variant: str) -> np.ndarray:
        return np.concatenate([probabilities[variant][stem] for stem in record.stems])

    table = []
    for variant in record.variants:
        rows = frame[frame["variant"] == variant]
        tokens = rows["tokens"].astype(int).tolist()
        layer = 0 if variant == "denso" else record.prune_layer
        row = {"Variante": variant_name(variant),
               CLASSES: class_agreement(pooled("denso"), pooled(variant))}
        if detections is not None:
            got = agreement(detections["denso"], detections[variant], tiou, min_score)
            row |= {RECALL: got["recall"], PRECISION: got["precision"],
                    MEAN_TIOU: got["mean_tiou"]}
        if scores is not None:
            row |= {f"mAP@{t:g}": value
                    for t, value in zip(scores[variant]["tiou"], scores[variant]["map"],
                                        strict=True)}
            row["mAP médio"] = scores[variant]["average"]
        if actions is not None:
            row |= task_metrics(record, frame, probabilities, detections, actions, variant,
                                tiou, min_score)
        row |= {
            "Tokens médios por clipe": np.mean(tokens),
            "Tokens mantidos (%)": np.mean(tokens) / GEOMETRY.num_tokens,
            # Dense keeps every token, so its share is the region's area.
            "Tokens na região do motorista": area if variant == "denso"
            else rows["inside_region"].mean(),
            "Cosseno médio com o denso": rows["cosine_to_dense"].mean(),
            "Cosseno P10 com o denso": rows["cosine_to_dense"].quantile(0.1),
            "Cosseno P50 com o denso": rows["cosine_to_dense"].median(),
        }
        if "rel_l2_to_dense" in rows:
            row["L2 relativo médio com o denso"] = rows["rel_l2_to_dense"].mean()
        row |= {
            "Speedup previsto": budget_cost(GEOMETRY, tokens, layer).predicted_speedup,
            "Speedup medido": dense_seconds / rows["seconds"].mean(),
            "Tempo médio por clipe (s)": rows["seconds"].mean(),
            "Tempo p50 por clipe (s)": rows["seconds"].median(),
            "Tempo p95 por clipe (s)": rows["seconds"].quantile(0.95),
        }
        if "reason" in rows and rows["reason"].notna().any():
            row[FALLBACK] = rows["reason"].isin(FALLBACKS).mean()
        fixed = _fixed(variant)
        if fixed and fixed[1] == "pa" and f"fixo_{fixed[0]}" in record.variants:
            base = frame[frame["variant"] == f"fixo_{fixed[0]}"]["seconds"].mean()
            row[PA_COST] = rows["seconds"].mean() / base - 1
        table.append(row)
    return pd.DataFrame(table)


def by_detection(record: RunRecord, frame: pd.DataFrame, probabilities: dict,
                 detections: dict | None, min_score: float,
                 reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each action of the reference read whole -- the dense pipeline's detections, or the
    annotated actions when given -- with the best tIoU each variant has with it, and the
    class the head gives each variant over the clips inside its window."""
    dense = reference is None
    reference = detections["denso"] if dense else reference
    reference = reference[reference["score"] >= min_score].reset_index(drop=True)
    table = pd.DataFrame({
        "Vídeo": reference["video-id"],
        "Classe": [f"classe {label}" for label in reference["label"]],
        "Início (s)": reference["t-start"],
        "Fim (s)": reference["t-end"],
    })
    if dense:
        table["Confiança do denso"] = reference["score"]
    for variant in record.variants:
        name = short_label(variant) if variant != "denso" else SHORT["denso"]
        if detections is not None:
            found = detections[variant]
            found = found[found["score"] >= min_score].reset_index(drop=True)
            best, where = _best(reference, found)
            table[f"{name}: melhor tIoU"] = best
            if not dense:
                table[f"{name}: confiança"] = [found.loc[at, "score"] if overlap > 0 else None
                                               for overlap, at in zip(best, where, strict=True)]
        classes = []
        for _, row in reference.iterrows():
            stem = row["video-id"]
            found_class = window_class(probabilities[variant][stem], _starts(frame, variant, stem),
                                       _span(record, stem), (row["t-start"], row["t-end"]))
            classes.append(None if found_class is None else f"classe {found_class}")
        table[f"{name}: classe no trecho"] = classes
    return table


def per_video_task(record: RunRecord, frame: pd.DataFrame, probabilities: dict,
                   detections: dict | None, actions: pd.DataFrame, scores: dict | None,
                   tiou: float, min_score: float) -> pd.DataFrame:
    """Each variant on each video, against that video's annotated actions, with its cost."""
    table = []
    for stem in record.stems:
        dense = frame[(frame["video"] == stem) & (frame["variant"] == "denso")]["seconds"]
        for variant in record.variants:
            rows = frame[(frame["video"] == stem) & (frame["variant"] == variant)]
            row = {"Vídeo": stem, "Variante": variant_name(variant),
                   "Ações anotadas": int((actions["video-id"] == stem).sum())}
            if scores is not None:
                row["mAP médio"] = scores[variant].get("by_video", {}).get(stem, {}).get("average")
            row |= task_metrics(record, frame, probabilities, detections, actions, variant, tiou,
                                min_score, [stem])
            row |= {"Cosseno P50 com o denso": rows["cosine_to_dense"].median(),
                    "Tempo médio por clipe (s)": rows["seconds"].mean(),
                    "Tempo p50 por clipe (s)": rows["seconds"].median(),
                    "Tempo p95 por clipe (s)": rows["seconds"].quantile(0.95),
                    "Tempo do vídeo (s)": rows["seconds"].sum(),
                    "Speedup medido": dense.mean() / rows["seconds"].mean()}
            table.append(row)
    return pd.DataFrame(table)


def per_class(record: RunRecord, frame: pd.DataFrame, probabilities: dict,
              detections: dict | None, actions: pd.DataFrame, tiou: float,
              min_score: float) -> pd.DataFrame:
    """Each class's annotated actions, per variant: how many are found again, how well and
    how confidently, and the class the head gives them. Few actions per class: read as
    counts, not as rates."""
    table = []
    for label, group in actions.groupby("label"):
        for variant in record.variants:
            row = {"Classe": f"classe {label}", "Variante": variant_name(variant),
                   "Ações": len(group)}
            if detections is not None:
                found, dense = detections[variant], detections["denso"]
                best = best_tiou(group, found[found["score"] >= min_score])
                common = shared_actions(group, found, dense, tiou, min_score)
                row |= {"Reencontradas": int((best >= tiou).sum()),
                        "tIoU médio do melhor par": float(best.mean()),
                        "Confiança média das reencontradas":
                            matched_confidence(common, found, tiou, min_score),
                        "Denso nas mesmas ações: confiança":
                            matched_confidence(common, dense, tiou, min_score)}
            head = annotated(record, frame, probabilities, variant, group)
            row["Acurácia de classe por ação (anotações)"] = head["action"]
            table.append(row)
    return pd.DataFrame(table)


def per_video(frame: pd.DataFrame, area: float) -> pd.DataFrame:
    levels = sorted(frame.loc[frame["variant"] != "denso", "tokens"].astype(int).unique())
    table = []
    for (video, variant), rows in frame.groupby(["video", "variant"], sort=False):
        row = {"Vídeo": video, "Variante": variant_name(variant), "Clipes": len(rows),
               "Tokens médios por clipe": rows["tokens"].mean()}
        for level in levels:
            row[f"Clipes com {level} tokens (%)"] = (
                None if variant == "denso" else (rows["tokens"] == level).mean())
        row["Tokens na região do motorista"] = (
            area if variant == "denso" else rows["inside_region"].mean())
        row["Cosseno médio com o denso"] = rows["cosine_to_dense"].mean()
        row["Tempo médio por clipe (s)"] = rows["seconds"].mean()
        table.append(row)
    return pd.DataFrame(table)


def fixed_mask_text(record: RunRecord) -> str:
    fixed = record.fixed_mask
    return f"{fixed['rows']} linhas do topo com peso {fixed['weight']}".replace(".", ",")


def configuration(record: RunRecord, policy: dict | None, region: dict, tiou: float,
                  min_score: float, actions: pd.DataFrame | None = None) -> pd.DataFrame:
    run, settings = record.run, record.settings
    hardware = run["environment"].get("gpu") or "CPU"
    masks = MASKS in record.variants
    items = [
        ("Modelo", "VideoMAEv2-g (ViT-g/14) com a cabeça de 16 classes do ajuste AI City"),
        ("Pesos", run["weights"]["path"]),
        ("Pesos: SHA-256", run["weights"]["sha256"]),
        ("Definição do modelo", run["model_source"]),
        ("Verificação do caminho denso", run["check"]),
        ("Ambiente", f"Python {run['environment']['python']}, torch "
                     f"{run['environment']['torch']}, {hardware} ({settings['device']}), "
                     f"precisão {settings['precision']}"),
    ]
    if policy is not None:
        fitted = policy["calibration"]
        items += [
            ("Política", f"{policy['policy']['version']} calibrada em {fitted['clips']} clipes de "
                         f"{len(fitted['videos'])} vídeos ({run['policy']['path']})"),
            ("Política: níveis e limiares",
             json.dumps({"levels": policy["policy"]["levels"],
                         "thresholds": policy["policy"]["thresholds"]})),
        ]
    items += [(f"Política {name}", f"{found['path']} (SHA-256 {found['sha256']})")
              for name, found in run.get("policies", {}).items()]
    items += [(f"Calibração de {name} com vídeos avaliados", json.dumps(found))
              for name, found in run.get("calibration_overlap", {}).items()
              if found["videos"] or found["drivers"]]
    if "probe_timing" in run:
        items.append(("Tempo do probe", json.dumps(run["probe_timing"])))
    if masks:
        found = region["calibration"]
        detector = found.get("detector") or {}
        detector_weights = Path(detector.get("weights", {}).get("path", "?")).name
        items += [
            ("Máscara fixa", fixed_mask_text(record)),
            ("Máscara inteligente",
             f"região do motorista: {found['area']:.0%} do quadro, motorista detectado em "
             f"{found['detected']} de {found['clips']} clipes de {len(found['videos'])} vídeos, "
             f"detector {detector_weights} (Ultralytics {detector.get('version', '?')}) "
             f"({run['region']['path']})"),
        ]
    items += [(f"Configuração: {k}", json.dumps(v, ensure_ascii=False))
              for k, v in settings.items()]
    items += [(f"{stem}: clipes", video["clips"]) for stem, video in run["schedules"].items()]
    items += [
        ("Nota: referência",
         ("Não há labels: a referência é o pipeline denso. " if actions is None else "")
         + "Cada variante é comparada com o pipeline denso na "
         "cabeça de classificação (a classe de maior probabilidade de cada clipe) e nas "
         f"detecções do AMA (as do denso com confiança de {min_score:g} para cima; uma "
         f"detecção da variante as reencontra quando é da mesma classe e tem tIoU de "
         f"{tiou:g} ou mais). Concordância não é acerto: se o denso errar, a variante que erra "
         "igual concorda. Mede o quanto a poda muda o resultado, não se ele está certo."),
    ]
    if actions is not None:
        items.append((
            "Nota: anotações",
            f"{len(actions)} ações anotadas nos vídeos avaliados (rótulos do AI City). O mAP "
            "vem do avaliador do próprio AMA; as ações reencontradas, a precisão, os falsos "
            f"positivos e as fronteiras usam as detecções com confiança de {min_score:g} para "
            f"cima e tIoU de {tiou:g} ou mais, na mesma classe. Fronteiras e confiança de uma "
            "variante são medidas nas ações que ela e o denso encontram, e as colunas 'Denso "
            "nas mesmas ações' dão o denso nesse mesmo conjunto. Esses vídeos estão no subset "
            "de treino do AMA, e o AMA viu recortes do motorista, não o quadro inteiro: os "
            "números medem o quanto cada variante preserva o desempenho do pipeline, não a "
            "generalização."))
    items += [
        ("Nota: drift", "Cosseno e L2 relativo com o embedding denso são métricas auxiliares: "
                        "uma variante pode se afastar do denso e reconhecer as ações, ou "
                        "ficar perto e perder o que importava."),
    ]
    if masks:
        items += [
            ("Nota: máscaras", "As três variantes do ViT-Prune usam a mesma política e mantêm os "
                               "mesmos tokens por clipe; as máscaras só mudam quais tokens "
                               "ficam."),
            ("Nota: calibração", "A política e a região vêm de vitprune calibrate e vitprune "
                                 "region, em vídeos que não são avaliados aqui."),
        ]
    else:
        masked = any(variant.endswith("_mascara") for variant in record.variants)
        items += [
            ("Nota: variantes", ("'+ máscara do motorista' só mantém tokens dentro da região do "
                                 "motorista da rodada; as demais variantes não usam máscara. "
                                 if masked else "Nenhuma variante usa máscara. ")
                                + "'Fixo N' mantém N tokens em todo "
                                "clipe; '+ atenção proporcional' é o mesmo orçamento com "
                                "size_biased_attention, e seu custo extra está na coluna "
                                f"'{PA_COST}'."),
        ]
    items += [
        ("Nota: pré-processamento", "Quadro inteiro (1920x1080) redimensionado para 224x224, "
                                    "como no pipeline do vitprune. O AMA e o ajuste do "
                                    "VideoMAEv2 usaram recortes do motorista (crop_videos): a "
                                    "acurácia absoluta do denso não é a do AI City; o contraste "
                                    "entre variantes é o que vale."),
        ("Nota: tempos", f"Medidos em {hardware}, {settings['batch']} clipe(s) por vez: "
                         "servem para comparar variantes entre si."
                         + ("" if masks else " A ordem das variantes alterna a cada lote, "
                            "para nenhuma ser sempre a primeira medida.")),
    ]
    return pd.DataFrame(items, columns=["Item", "Valor"])


def writable(path: Path) -> Path:
    """The path itself, or a timestamped sibling when another program holds it open:
    Windows locks a workbook open in Excel and an image open in some viewers."""
    try:
        path.open("ab").close()
        return path
    except OSError:
        found = path.with_stem(f"{path.stem}-{datetime.now():%Y%m%d-%H%M%S}")
        print(f"{path.name} is open elsewhere; writing {found.name} instead")
        return found


def write_excel(record: RunRecord, sheets: dict[str, pd.DataFrame]) -> Path:
    path = writable(record.out / "benchmark-vitprune.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            for column, title in enumerate(table.columns, start=1):
                width = max([len(str(title)), *(len(f"{v:.3f}" if isinstance(v, float) else str(v))
                                                for v in table[title].head(200))])
                letter = sheet.cell(row=1, column=column).column_letter
                sheet.column_dimensions[letter].width = min(width + 2, 70)
                share = any(k in str(title) for k in PERCENT)
                for row in range(2, len(table) + 2):
                    cell = sheet.cell(row=row, column=column)
                    if isinstance(cell.value, float):
                        cell.number_format = "0.0%" if share else "0.000"
            sheet.freeze_panes = "A2"
    print(f"written {path}")
    return path


def cells_to_pixels(cells: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height) * GRID // height
    columns = np.arange(width) * GRID // width
    return cells.reshape(GRID, GRID)[rows[:, None], columns[None, :]]


def shade(image: np.ndarray, inside: np.ndarray, dim: float = 0.3) -> np.ndarray:
    out = image.astype(np.float32)
    out[~cells_to_pixels(inside, *image.shape[:2])] *= dim
    return out.clip(0, 255).astype(np.uint8)


def outline(axis, cells: np.ndarray, height: int, width: int, color: str) -> None:
    for position in np.flatnonzero(cells.reshape(-1)):
        row, column = divmod(int(position), GRID)
        axis.add_patch(Rectangle((column * width / GRID, row * height / GRID), width / GRID,
                                 height / GRID, fill=False, edgecolor=color, linewidth=0.6))


def frame_of(record: RunRecord, stem: str, start: int) -> np.ndarray:
    image = plt.imread(record.frames / stem / f"{start:06d}.jpg")
    return image[::2, ::2]


def show_fixed(axis, image: np.ndarray, fixed: np.ndarray) -> None:
    height, width = image.shape[:2]
    axis.imshow(shade(image, fixed > 0.5, dim=0.35))
    outline(axis, fixed < 0.5, height, width, "orangered")


def show_smart(axis, image: np.ndarray, smart: np.ndarray) -> None:
    height, width = image.shape[:2]
    axis.imshow(shade(image, smart > 0.5))
    outline(axis, smart > 0.5, height, width, "magenta")


def masks_figure(record: RunRecord, stem: str, start: int, smart: np.ndarray,
                 fixed: np.ndarray) -> None:
    image = frame_of(record, stem, start)
    figure, axes = plt.subplots(1, 2, figsize=(16, 4.9))
    show_fixed(axes[0], image, fixed)
    axes[0].set_title(f"Máscara fixa: {fixed_mask_text(record)}")
    show_smart(axes[1], image, smart)
    axes[1].set_title(f"Máscara inteligente (YOLO nos vídeos de calibração): "
                      f"{smart.mean():.0%} do quadro")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"Máscaras aplicadas a {stem}", fontsize=13)
    figure.tight_layout()
    figure.savefig(writable(record.figures / f"mascaras-{stem}.png"), dpi=110)
    plt.close(figure)


def selection_figure(record: RunRecord, frame: pd.DataFrame, stem: str, start: int) -> None:
    arrays = np.load(record.out / "selection" / f"{stem}_{start:06d}.npz")
    rows = frame[(frame["video"] == stem) & (frame["start_frame"] == start)].set_index("variant")
    image = frame_of(record, stem, start)
    height, width = image.shape[:2]

    figure, axes = plt.subplots(2, 3, figsize=(19, 8.4))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Quadro original", fontsize=10)
    show_fixed(axes[0, 1], image, arrays["vitprune_mascara_fixa_prior"])
    axes[0, 1].set_title(f"Máscara fixa ({fixed_mask_text(record)})", fontsize=10)
    smart = arrays["vitprune_mascara_inteligente_prior"]
    show_smart(axes[0, 2], image, smart)
    axes[0, 2].set_title(f"Máscara inteligente ({smart.mean():.0%} do quadro)", fontsize=10)
    for axis, variant in zip(axes[1], PRUNED, strict=True):
        kept = arrays[f"{variant}_kept"][MIDDLE_SLICE]
        axis.imshow(shade(image, kept, dim=0.25))
        outline(axis, kept, height, width, "lime")
        row = rows.loc[variant]
        axis.set_title(f"{SHORT[variant]}: {kept.sum()} de {GRID * GRID} tokens na fatia "
                       f"({int(row['tokens'])} no clipe)\n"
                       f"{row['inside_region']:.0%} dos tokens na região do motorista",
                       fontsize=10)
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(f"{stem} | clipe em {rows['start_s'].iloc[0]:.1f} s  (claro = token "
                    f"mantido, escuro = fundido; fatia temporal central)", fontsize=12)
    figure.tight_layout()
    figure.savefig(writable(record.figures / f"selecao-{stem}-{start:06d}.png"), dpi=100)
    plt.close(figure)


def timeline_figure(record: RunRecord, stem: str, detections: dict[str, pd.DataFrame],
                    min_score: float = MIN_SCORE, actions: pd.DataFrame | None = None) -> None:
    """What the AMA found along the video, per variant, under the annotated actions when
    there are any; the dense one is the reference."""
    lanes = {} if actions is None else {"Anotações": actions}
    lanes |= {short_label(variant) if variant == "denso" else f"AMA: {short_label(variant)}":
              detections[variant] for variant in record.variants}
    ids = sorted({int(label) for found in lanes.values() for label in found["label"]})
    colors = {label: plt.cm.tab20.colors[label % 20] for label in ids}
    lanes = [(title, list(zip(found["t-start"], found["t-end"], found["label"], found["score"],
                              strict=True)))
             for title, found in ((title, found[(found["video-id"] == stem)
                                                & (found["score"] >= min_score)])
                                  for title, found in lanes.items())]
    figure, axis = plt.subplots(figsize=(22, 1.2 + 0.6 * len(lanes)))
    for lane, (_, bars) in enumerate(lanes):
        for start, end, label, score in sorted(bars, key=lambda bar: bar[3]):
            axis.barh(lane, end - start, left=start, height=0.7, color=colors[int(label)],
                      alpha=0.35 + 0.65 * min(float(score), 1.0))
    axis.set_yticks(range(len(lanes)), [title for title, _ in lanes])
    axis.invert_yaxis()
    axis.set_xlabel("segundos no vídeo")
    axis.set_title(f"{stem}: detecções do AMA (confiança ≥ {min_score:g}; "
                   "mais opaco = mais confiante)")
    if ids:
        figure.legend(handles=[Patch(color=colors[i], label=f"classe {i}") for i in ids],
                      loc="lower center", ncol=8, fontsize=8)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(writable(record.figures / f"linha-do-tempo-{stem}.png"), dpi=110)
    plt.close(figure)


def summary_figure(record: RunRecord, table: pd.DataFrame) -> None:
    variants = record.variants
    metrics = [
        ("mAP médio", "mAP médio contra as anotações (%)", 100, "{:.1f}"),
        (GT_RECALL, "Ações anotadas reencontradas (%)", 100, "{:.0f}"),
        (RECALL, "Detecções do denso reencontradas (%)", 100, "{:.0f}"),
        (CLASSES, "Concordância de classe com o denso (%)", 100, "{:.0f}"),
        ("Tokens na região do motorista", "Tokens na região do motorista (%)", 100, "{:.0f}"),
        ("Tokens médios por clipe", "Tokens médios por clipe", 1, "{:.0f}"),
        ("Speedup medido", "Speedup medido", 1, "{:.1f}x"),
    ]
    metrics = [m for m in metrics if m[0] in table and table[m[0]].notna().any()]
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.6))
    for axis, (column, title, scale, label) in zip(np.atleast_1d(axes), metrics, strict=True):
        values = (table[column] * scale).tolist()
        axis.bar(range(len(variants)), values,
                 color=[color(v, i) for i, v in enumerate(variants)])
        for x, value in enumerate(values):
            if value == value:  # skips NaN
                axis.text(x, value, label.format(value), ha="center", va="bottom", fontsize=8)
        axis.set_xticks(range(len(variants)), [short_label(v) for v in variants], rotation=20,
                        fontsize=8)
        axis.set_title(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(writable(record.figures / "resumo.png"), dpi=110)
    plt.close(figure)


def quality_cost_figure(record: RunRecord, table: pd.DataFrame) -> None:
    """Quality against the time measured per clip, one point per variant: the trade the
    budgets make. The cosine to the dense embedding is beside the task, never instead of it."""
    panels = [("Cosseno P10 com o denso", "cosseno P10 com o denso (auxiliar)")]
    if "mAP médio" in table:
        panels.insert(0, ("mAP médio", "mAP médio contra as anotações"))
    figure, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.8))
    for axis, (column, title) in zip(np.atleast_1d(axes), panels, strict=True):
        for index, (variant, (_, row)) in enumerate(zip(record.variants, table.iterrows(),
                                                        strict=True)):
            axis.scatter(row["Tempo médio por clipe (s)"], row[column], s=60,
                         color=color(variant, index))
            axis.annotate(short_label(variant), (row["Tempo médio por clipe (s)"], row[column]),
                          textcoords="offset points", xytext=(5, 5), fontsize=8)
        axis.set_xlabel("tempo medido por clipe (s)")
        axis.set_title(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(writable(record.figures / "qualidade-custo.png"), dpi=110)
    plt.close(figure)


def make_report(out: str | Path, min_score: float = MIN_SCORE, tiou: float = TIOU) -> Path:
    """Write the workbook and the figures under ``out``, the folder of a finished run.

    Args:
        out: the results folder of ``vitprune benchmark run``.
        min_score: detections below this confidence are left out, in the dense reference and
            in each variant.
        tiou: the overlap at which a variant's detection matches one of the dense, or an
            annotated action.

    Returns:
        The workbook.
    """
    record = RunRecord.read(out)
    try:
        import openpyxl  # noqa: F401 -- pandas writes the workbook through it
    except ImportError as error:  # better now than after the tables are built
        raise RuntimeError(
            "the workbook needs openpyxl: pip install 'vitprune[benchmark]'"
        ) from error
    frame = load(record)
    policy, region = recorded(record)
    area = float(np.mean(region["region"]["prior"]))
    detections = ama_detections(record)
    actions = ground_truth(record)
    scores = ama_scores(record)
    probabilities = head_probabilities(record, frame)
    table = summary(record, frame, probabilities, detections, area, tiou, min_score, actions,
                    scores)
    sheets = {"Resumo": table}
    if actions is not None:
        sheets["Por vídeo e variante"] = per_video_task(record, frame, probabilities, detections,
                                                        actions, scores, tiou, min_score)
        sheets["Por classe"] = per_class(record, frame, probabilities, detections, actions, tiou,
                                         min_score)
        sheets["Por ação anotada"] = by_detection(record, frame, probabilities, detections,
                                                  min_score, reference=actions)
    if detections is not None:
        sheets["Por detecção do denso"] = by_detection(record, frame, probabilities, detections,
                                                       min_score)
    sheets["Por vídeo"] = per_video(frame, area)
    sheets["Configuração"] = configuration(record, policy, region, tiou, min_score, actions)
    workbook = write_excel(record, sheets)

    record.figures.mkdir(parents=True, exist_ok=True)
    summary_figure(record, table)
    quality_cost_figure(record, table)
    shown = sorted((record.out / "selection").glob("*.npz")) if MASKS in record.variants else []
    for path in shown:
        stem, start = path.stem.rsplit("_", 1)
        selection_figure(record, frame, stem, int(start))
    for stem in record.stems:
        first = next((p for p in shown if p.stem.rsplit("_", 1)[0] == stem), None)
        if first is not None:
            arrays = np.load(first)
            masks_figure(record, stem, int(first.stem.rsplit("_", 1)[1]),
                         arrays["vitprune_mascara_inteligente_prior"],
                         arrays["vitprune_mascara_fixa_prior"])
        if detections is not None:
            timeline_figure(record, stem, detections, min_score, actions)
    print(f"{len(frame) // len(record.variants)} clips, figures in {record.figures}")
    return workbook
