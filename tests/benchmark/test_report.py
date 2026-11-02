"""The code that turns benchmark runs into the thesis tables: where the kept tokens fall,
which detection answers each action, the class of each action read whole, and what the
AMA is scored against.

Skipped without the benchmark's own dependencies (pandas, matplotlib).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))

import evaluate_ama  # noqa: E402
import make_report  # noqa: E402
import run_inference  # noqa: E402
from common import VARIANTS  # noqa: E402


def test_the_share_inside_the_region_ignores_the_padding():
    region = torch.tensor([1.0, 1.0, 0.0, 0.0])
    kept = torch.tensor([
        [[0, 2], [1, 3]],  # two of four kept tokens inside
        [[0, -1], [1, -1]],  # one per slice, padded to the widest clip; both inside
    ])
    assert run_inference.inside_share(kept, region) == [0.5, 1.0]


def _detections(rows):
    return pd.DataFrame(rows, columns=["video-id", "t-start", "t-end", "label", "score"])


def test_an_action_is_answered_by_the_most_confident_detection_of_its_class():
    detections = _detections([
        ["v", 10.0, 20.0, 3, 0.4],  # aligned, less confident
        ["v", 15.0, 25.0, 3, 0.9],  # half overlapping, the most confident
        ["v", 10.0, 20.0, 5, 0.99],  # another class
        ["w", 10.0, 20.0, 3, 0.95],  # another video
        ["v", 30.0, 40.0, 3, 0.97],  # elsewhere
    ])
    tiou, score = make_report.match(detections, "v", {"label_id": 3, "start": 10.0, "end": 20.0})
    assert score == 0.9
    assert tiou == pytest.approx(5 / 15)


def test_an_action_nothing_overlaps_is_not_answered():
    detections = _detections([["v", 30.0, 40.0, 3, 0.9]])
    assert make_report.match(detections, "v", {"label_id": 3, "start": 0.0, "end": 10.0}) is None


def test_each_action_is_classified_from_all_its_clips(tmp_path, monkeypatch):
    """Rows arrive in any order; the features are in clip order and must line up."""
    annotated = [{"label_id": 1, "label": "Drinking", "start": 0.0, "end": 2.0},
                 {"label_id": 2, "label": "Eating", "start": 2.0, "end": 4.0}]
    monkeypatch.setattr(make_report, "RESULTS", tmp_path)
    monkeypatch.setattr(make_report, "videos", lambda: [Path("v.mp4")])
    monkeypatch.setattr(make_report, "segments", lambda stem: annotated)
    monkeypatch.setattr(make_report, "class_names", lambda: {0: "Other", 1: "Drinking",
                                                              2: "Eating"})
    starts, owners = [0, 16, 32, 48], [0, 0, 1, None]
    right = np.array([[0.1, 0.8, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [1.0, 0.0, 0.0]])
    wrong = right[:, [1, 0, 2]]  # the first action now reads as "Other"
    rows = []
    for variant in VARIANTS:
        probabilities = wrong if variant == "vitprune" else right
        (tmp_path / "features" / variant).mkdir(parents=True)
        np.savez(tmp_path / "features" / variant / "v.npz", a_prob=probabilities)
        rows += [{"video": "v", "variant": variant, "start_frame": starts[i],
                  "segment": owners[i], "tokens": 256 * (i + 1)} for i in (2, 0, 3, 1)]
    detections = {variant: _detections([["v", 0.0, 2.0, 1, 0.8]]) for variant in VARIANTS}

    table = make_report.actions(pd.DataFrame(rows), detections).set_index("Ação")

    assert table.loc["Drinking", "Denso: classe no trecho"] == "Drinking"
    assert table.loc["Drinking", "Denso: acerto"]
    assert not table.loc["Drinking", "Sem máscara: acerto"]
    assert table.loc["Eating", "Sem máscara: acerto"]
    assert table.loc["Drinking", "Clipes no trecho"] == 2
    assert table.loc["Drinking", "Tokens médios (ViT-Prune)"] == 384
    assert table.loc["Drinking", "Denso: AMA tIoU"] == 1.0
    assert table.loc["Eating", "Denso: AMA tIoU"] == 0.0


def test_the_ama_is_scored_on_the_evaluated_videos_only(tmp_path, monkeypatch):
    """The AMA skips a video without features but would count its actions as missed."""
    database = {"database": {"evaluated": {"annotations": []}, "other": {"annotations": []}}}
    monkeypatch.setattr(evaluate_ama, "annotations", lambda: database)
    monkeypatch.setattr(evaluate_ama, "RESULTS", tmp_path)
    monkeypatch.setitem(evaluate_ama.SETTINGS, "evaluate", ["evaluated"])

    written = json.loads(evaluate_ama.labels().read_text(encoding="utf-8"))["database"]

    assert written == {"evaluated": {"annotations": [], "subset": "validation"}}


def test_a_run_of_another_configuration_is_not_resumed(tmp_path, monkeypatch):
    policy, region = tmp_path / "policy.json", tmp_path / "region.json"
    policy.write_text("{}", encoding="utf-8")
    region.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_inference, "POLICY", policy)
    monkeypatch.setattr(run_inference, "REGION", region)
    record = tmp_path / "run.json"
    settings = {"stride": 16, "batch": 8}

    run_inference.refuse_other_runs(record, settings)  # nothing to resume yet
    record.write_text(json.dumps({
        "settings": settings,
        "policy": {"sha256": run_inference.SourceFile.of(policy).sha256},
        "region": {"sha256": run_inference.SourceFile.of(region).sha256},
    }), encoding="utf-8")
    run_inference.refuse_other_runs(record, settings)
    with pytest.raises(SystemExit, match="another configuration"):
        run_inference.refuse_other_runs(record, {**settings, "batch": 4})
    region.write_text('{"other": 1}', encoding="utf-8")
    with pytest.raises(SystemExit, match="another configuration"):
        run_inference.refuse_other_runs(record, settings)
