"""The report, which reads every variant against the dense one: how much of the dense
pipeline's answer each variant keeps in the classification head and in the AMA's detections.

Skipped without the report's own dependencies (pandas, matplotlib); the tests that write
the workbook also need openpyxl.
"""

import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from matplotlib import pyplot as plt  # noqa: E402

from vitprune.pipeline.benchmark import report  # noqa: E402
from vitprune.pipeline.benchmark.settings import VARIANTS  # noqa: E402
from vitprune.pipeline.cli import main  # noqa: E402

DENSE, FIXED = "Sem ViT-Prune (denso, referência)", "ViT-Prune + máscara fixa"
RECALL = "Detecções do denso reencontradas (recall)"
CLASSES = "Concordância de classe com o denso"


def _d(rows):
    return pd.DataFrame(rows, columns=["video-id", "t-start", "t-end", "label", "score"])


def test_best_tiou_pairs_each_detection_with_its_best_same_class_overlap():
    reference = _d([["v", 10, 20, 3, .9], ["v", 30, 40, 3, .9], ["w", 0, 5, 3, .9]])
    found = _d([["v", 15, 25, 3, .4], ["v", 10, 20, 5, .9], ["v", 12, 20, 3, .1]])

    assert report.best_tiou(reference, found) == pytest.approx([0.8, 0.0, 0.0])


def test_agreement_counts_what_the_variant_finds_and_what_it_invents():
    reference = _d([["v", 10, 20, 3, .9], ["v", 30, 40, 3, .9],
                    ["v", 10, 20, 3, .1]])  # the last is below the floor
    found = _d([["v", 10, 21, 3, .8], ["v", 50, 60, 2, .7]])

    got = report.agreement(reference, found, tiou=0.5, min_score=0.3)

    assert (got["recall"], got["precision"]) == (0.5, 0.5)
    assert got["mean_tiou"] == pytest.approx((10 / 11) / 2)


def test_a_variant_that_finds_nothing_has_zero_recall_and_no_precision():
    got = report.agreement(_d([["v", 10, 20, 3, .9]]), _d([]))
    assert (got["recall"], got["precision"], got["mean_tiou"]) == (0.0, None, 0.0)


def test_a_dense_run_without_detections_leaves_recall_undefined():
    got = report.agreement(_d([]), _d([["v", 10, 20, 3, .9]]))
    assert (got["recall"], got["mean_tiou"], got["precision"]) == (None, None, 0.0)


def test_class_agreement_is_the_share_of_clips_with_the_dense_class():
    dense = np.eye(3)[[0, 1, 2, 0]]

    assert report.class_agreement(dense, np.eye(3)[[0, 1, 1, 0]]) == 0.75
    with pytest.raises(ValueError, match="clips"):
        report.class_agreement(dense, dense[:3])


def test_the_class_in_a_window_averages_the_clips_wholly_inside_it():
    # Inside (0, 4): the first three clips, averaging to [1/3, 2/3]. The last one ends at 5 s;
    # counted, it would tie the average at [1/2, 1/2] and give class 0.
    probabilities = np.array([[.8, .2], [.1, .9], [.1, .9], [1., 0.]])
    starts = np.array([0., 1., 2., 3.])  # clips of 2 s

    assert report.window_class(probabilities, starts, 2.0, (0.0, 4.0)) == 1
    assert report.window_class(probabilities, starts, 2.0, (10.0, 12.0)) is None


def _synthetic_run(tmp_path, with_ama=True, dense_detections=True, stems=("v",)):
    """A finished run of 4 clips of each video, with the dense and the fixed-mask variants
    disagreeing on one clip and on one detection."""
    out = tmp_path / "run"
    grid, slices = report.GRID, report.GEOMETRY.temporal_slices
    rng = np.random.default_rng(0)
    starts, rows = [0, 16, 32, 48], []
    for variant in VARIANTS:
        (out / "features" / variant).mkdir(parents=True)
        for stem in stems:
            for start in starts:
                rows.append({"video": stem, "start_frame": start, "start_s": start / 8,
                             "variant": variant, "tokens": 2048 if variant == "denso" else 512,
                             "regime": None if variant == "denso" else 1,
                             "inside_region": None if variant == "denso" else 0.8,
                             "cosine_to_dense": 0.95,
                             "seconds": 1.0 if variant == "denso" else 0.4})
            classes = [0, 1, 0, 2] if variant == "vitprune_mascara_fixa" else [0, 1, 1, 2]
            np.savez(out / "features" / variant / f"{stem}.npz",
                     a_prob=np.eye(3)[classes] * 0.8 + 0.1, feats=np.zeros((4, 6)))
    (out / "clips.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    smart = np.zeros((grid, grid))
    smart[4:, 1:14] = 1
    fixed = np.ones((grid, grid))
    fixed[:4] = 0.15
    (out / "selection").mkdir()
    for stem in stems:
        (out / "frames" / stem).mkdir(parents=True)
        for start in (0, 48):
            arrays = {f"{v}_kept": rng.random((slices, grid * grid)) < 0.25 for v in VARIANTS[1:]}
            arrays["vitprune_mascara_fixa_prior"] = fixed.flatten()
            arrays["vitprune_mascara_inteligente_prior"] = smart.flatten()
            np.savez_compressed(out / "selection" / f"{stem}_{start:06d}.npz", **arrays)
            plt.imsave(out / "frames" / stem / f"{start:06d}.jpg", rng.random((108, 192, 3)))

    if with_ama:
        def both(stem):
            return [[stem, 0.0, 10.0, 1, .9], [stem, 4.0, 8.0, 2, .8]]

        for variant in VARIANTS:
            lines = [line for stem in stems for line in (
                both(stem)[:1] if variant == "vitprune_mascara_fixa"
                else [] if variant == "denso" and not dense_detections else both(stem))]
            (out / "ama" / variant).mkdir(parents=True)
            _d(lines).to_csv(out / "ama" / variant / "predictions.csv", index=False)

    (out / "policy.json").write_text(json.dumps({
        "policy": {"version": "p1", "levels": [256, 512, 768], "thresholds": [0.3, 0.6]},
        "calibration": {"clips": 100, "videos": [{}, {}]},
    }), encoding="utf-8")
    (out / "region.json").write_text(json.dumps({
        "region": {"grid_size": grid, "prior": smart.flatten().tolist()},
        "calibration": {"area": float(smart.mean()), "detected": 90, "clips": 100,
                        "videos": [{}, {}],
                        "detector": {"version": "8.4", "weights": {"path": "/x/yolo26s.pt"}}},
    }), encoding="utf-8")
    (out / "run.json").write_text(json.dumps({
        "check": "dense path matches", "weights": {"path": "/x/w.pth", "sha256": "abc"},
        "model_source": "/x/src", "policy": {"path": str(out / "policy.json")},
        "region": {"path": str(out / "region.json")},
        "environment": {"python": "3.10", "torch": "2.x", "gpu": "Orin"},
        "settings": {"device": "cuda", "precision": "off", "batch": 8, "prune_layer": 2,
                     "stride": 16, "frame_step": 2, "fixed_mask": {"rows": 4, "weight": 0.15}},
        "schedules": {stem: {"fps": 8.0, "frames": 64, "clips": 4} for stem in stems},
    }), encoding="utf-8")
    return out


def test_the_report_without_labels_compares_every_variant_with_the_dense(tmp_path):
    pytest.importorskip("openpyxl")
    out = _synthetic_run(tmp_path)

    book = pd.ExcelFile(report.make_report(out))

    assert book.sheet_names == ["Resumo", "Por detecção do denso", "Por vídeo", "Configuração"]
    summary = book.parse("Resumo").set_index("Variante")
    assert summary.loc[DENSE, RECALL] == 1.0 and summary.loc[DENSE, CLASSES] == 1.0
    assert summary.loc[FIXED, RECALL] == 0.5 and summary.loc[FIXED, CLASSES] == 0.75
    assert summary.loc[FIXED, "Speedup medido"] == pytest.approx(2.5)
    columns = [str(c) for sheet in book.sheet_names for c in book.parse(sheet).columns]
    assert not any("acerto" in c.lower() or "mAP" in c for c in columns)
    detections = book.parse("Por detecção do denso")
    assert detections["Classe"].tolist() == ["classe 1", "classe 2"]
    assert detections["Máscara fixa: melhor tIoU"].tolist() == [1.0, 0.0]
    notes = book.parse("Configuração").set_index("Item")["Valor"]
    assert "acerto" in notes["Nota: referência"]  # says what the agreement is not
    figures = out / "figuras"
    assert (figures / "resumo.png").exists() and (figures / "linha-do-tempo-v.png").exists()
    assert (figures / "mascaras-v.png").exists()
    assert len(list(figures.glob("selecao-*.png"))) == 2  # clips 0 and 48 of the one video


def test_the_report_without_the_ama_still_writes_the_head_and_time_columns(tmp_path):
    pytest.importorskip("openpyxl")
    out = _synthetic_run(tmp_path, with_ama=False)

    summary = pd.ExcelFile(report.make_report(out)).parse("Resumo")

    assert CLASSES in summary.columns and "Speedup medido" in summary.columns
    assert RECALL not in summary.columns
    assert not (out / "figuras" / "linha-do-tempo-v.png").exists()


def test_a_dense_run_without_detections_still_gives_a_report(tmp_path):
    pytest.importorskip("openpyxl")
    out = _synthetic_run(tmp_path, dense_detections=False)

    book = pd.ExcelFile(report.make_report(out))

    assert book.parse("Por detecção do denso").empty
    assert book.parse("Resumo").set_index("Variante").loc[DENSE, CLASSES] == 1.0


def test_the_report_needs_a_finished_run(tmp_path, capsys):
    assert main(["benchmark", "report", "--out", str(tmp_path)]) == 2
    assert "vitprune benchmark run" in capsys.readouterr().err


def test_the_report_step_takes_its_thresholds_from_its_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        report, "make_report",
        lambda out, min_score, tiou: seen.update(out=out, min_score=min_score, tiou=tiou),
    )

    assert main(["benchmark", "report", "--out", "o", "--min-score", "0.4", "--tiou", "0.6"]) == 0

    assert seen == {"out": "o", "min_score": 0.4, "tiou": 0.6}


def test_the_selection_figures_of_videos_with_the_same_name_ending_do_not_overwrite_each_other(
    tmp_path,
):
    pytest.importorskip("openpyxl")
    out = _synthetic_run(tmp_path, stems=("Dashboard_NoAudio_5", "Other_NoAudio_5"))

    report.make_report(out)

    names = sorted(p.name for p in (out / "figuras").glob("selecao-*.png"))
    assert names == ["selecao-Dashboard_NoAudio_5-000000.png",
                     "selecao-Dashboard_NoAudio_5-000048.png",
                     "selecao-Other_NoAudio_5-000000.png",
                     "selecao-Other_NoAudio_5-000048.png"]


def test_the_dense_row_says_it_is_the_reference():
    assert report.NAMES["denso"] == DENSE
    assert report.short_label("denso") == "Denso (referência)"
    assert report.short_label("vitprune") == "Sem máscara"


# ---------------------------------------------------------------------------
# Against the annotations, and the policy study
# ---------------------------------------------------------------------------
def test_boundary_errors_measure_the_matched_actions_only():
    actions = _d([["v", 0.0, 10.0, 1, 1.0], ["v", 20.0, 30.0, 2, 1.0]])
    found = _d([["v", 1.0, 9.0, 1, 0.9]])
    assert report.boundary_errors(actions, found) == {"matched": 1, "start": 1.0, "end": 1.0}
    assert report.boundary_errors(actions, _d([]))["matched"] == 0


def test_false_positives_are_detections_matching_no_action():
    actions = _d([["v", 0.0, 10.0, 1, 1.0]])
    found = _d([["v", 1.0, 9.0, 1, 0.9], ["v", 40.0, 50.0, 1, 0.8], ["v", 0.0, 10.0, 2, 0.2]])
    assert report.false_positives(actions, found) == 1  # the last is below the minimum score


def test_clip_accuracy_reads_the_action_holding_each_clip_centre():
    probabilities = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.8, .1, .1]])
    actions = _d([["v", 2.0, 6.0, 1, 1.0]])
    assert report.clip_accuracy(probabilities, np.array([1.0, 3.0, 5.0, 7.0]), actions) == (1, 2)


def _policy_run(tmp_path):
    """A finished policy study of 4 clips of one video, with the AMA and the annotations:
    the fixed budget finds one of the two annotated actions, the dense one both."""
    out = tmp_path / "run"
    variants = ("denso", "fixo_32", "fixo_32_pa")
    seconds = {"denso": 1.0, "fixo_32": 0.4, "fixo_32_pa": 0.5}
    rows = []
    for variant in variants:
        (out / "features" / variant).mkdir(parents=True)
        for start in (0, 16, 32, 48):
            rows.append({"video": "v", "start_frame": start, "start_s": start / 8,
                         "variant": variant, "tokens": 64 if variant == "denso" else 32,
                         "regime": None, "inside_region": None,
                         "cosine_to_dense": 1.0 if variant == "denso" else 0.8,
                         "rel_l2_to_dense": 0.0 if variant == "denso" else 0.5,
                         "predicted_risk": None, "reason": None, "seconds": seconds[variant]})
        np.savez(out / "features" / variant / "v.npz", a_prob=np.eye(3)[[0, 1, 1, 2]] * 0.8 + 0.1,
                 feats=np.zeros((4, 6)))
        (out / "ama" / variant).mkdir(parents=True)
        found = [["v", 0.0, 10.0, 1, .9]] + ([["v", 4.0, 8.0, 2, .8]] if variant == "denso" else [])
        _d(found).to_csv(out / "ama" / variant / "predictions.csv", index=False)
        average = 0.6 if variant == "denso" else 0.3
        (out / "ama" / variant / "map.json").write_text(json.dumps(
            {"tiou": [0.1, 0.2, 0.3, 0.4, 0.5], "map": [average] * 5, "average": average,
             "by_video": {"v": {"map": [average] * 5, "average": average}}}))
    (out / "clips.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (out / "ama" / "labels.json").write_text(json.dumps({"database": {"v": {
        "subset": "validation", "annotations": [
            {"label_id": 1, "label": "a", "segment": [0.0, 10.0]},
            {"label_id": 2, "label": "b", "segment": [4.0, 8.0]}]}}}), encoding="utf-8")
    grid = report.GRID
    (out / "region.json").write_text(json.dumps({
        "region": {"grid_size": grid, "prior": [1.0] * (grid * grid)},
        "calibration": {"area": 1.0, "detected": 1, "clips": 1, "videos": [{}]}}), "utf-8")
    (out / "run.json").write_text(json.dumps({
        "check": "dense path matches", "weights": {"path": "/x/w.pth", "sha256": "abc"},
        "model_source": "/x/src", "policy": None, "policies": {},
        "region": {"path": str(out / "region.json")},
        "environment": {"python": "3.10", "torch": "2.x", "gpu": "Orin"},
        "settings": {"device": "cuda", "precision": "off", "batch": 8, "prune_layer": 2,
                     "stride": 16, "frame_step": 2, "fixed_mask": {"rows": 4, "weight": 0.15},
                     "variants": list(variants)},
        "schedules": {"v": {"fps": 8.0, "frames": 64, "clips": 4}},
    }), encoding="utf-8")
    return out


def test_the_policy_study_report_separates_fidelity_agreement_and_annotations(tmp_path):
    pytest.importorskip("openpyxl")
    out = _policy_run(tmp_path)

    book = pd.ExcelFile(report.make_report(out))

    assert "Por ação anotada" in book.sheet_names
    summary = book.parse("Resumo").set_index("Variante")
    dense, fixed, proportional = (report.variant_name(v) for v in ("denso", "fixo_32",
                                                                    "fixo_32_pa"))
    assert summary.loc[dense, "mAP médio"] == pytest.approx(0.6)
    assert summary.loc[fixed, "mAP médio"] == pytest.approx(0.3)
    assert summary.loc[dense, report.GT_RECALL] == 1.0
    assert summary.loc[fixed, report.GT_RECALL] == 0.5
    assert summary.loc[fixed, "Cosseno P10 com o denso"] == pytest.approx(0.8)
    assert summary.loc[proportional, report.PA_COST] == pytest.approx(0.25)
    assert pd.isna(summary.loc[fixed, report.PA_COST])
    actions = book.parse("Por ação anotada")
    assert actions["Classe"].tolist() == ["classe 1", "classe 2"]
    figures = out / "figuras"
    assert (figures / "qualidade-custo.png").exists()
    assert not list(figures.glob("mascaras-*.png"))


def test_the_task_is_also_reported_per_video_and_per_class(tmp_path):
    pytest.importorskip("openpyxl")
    out = _policy_run(tmp_path)

    book = pd.ExcelFile(report.make_report(out))

    videos = book.parse("Por vídeo e variante").set_index("Variante")
    fixed = report.variant_name("fixo_32")
    assert videos.loc[fixed, "mAP médio"] == pytest.approx(0.3)
    assert videos.loc[fixed, "Ações anotadas"] == 2
    assert videos.loc[fixed, report.GT_RECALL] == 0.5
    assert videos.loc[fixed, "Tempo do vídeo (s)"] == pytest.approx(1.6)  # 4 clips of 0.4 s
    classes = book.parse("Por classe")
    second = classes[(classes["Classe"] == "classe 2") & (classes["Variante"] == fixed)]
    assert second["Reencontradas"].tolist() == [0]
    dense = classes[(classes["Classe"] == "classe 2")
                    & (classes["Variante"] == report.variant_name("denso"))]
    assert dense["Confiança média das reencontradas"].tolist() == [pytest.approx(0.8)]
    actions = book.parse("Por ação anotada")
    assert actions["Denso: confiança"].tolist() == [pytest.approx(0.9), pytest.approx(0.8)]


def test_a_masked_budget_is_named_for_its_mask():
    assert report.variant_name("fixo_768_mascara") == (
        "ViT-Prune fixo 768 tokens + máscara do motorista")
    assert report.short_label("fixo_768_mascara") == "Fixo 768 + máscara"
