"""The AMA step, on a fake AMA project: a minimal ``libs`` package with the same entry points
the real one has, so the wiring is exercised without a GPU, weights or the real model.
"""

import ast
import csv
import hashlib
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from vitprune.pipeline.benchmark import ama
from vitprune.pipeline.benchmark.ama import run_ama
from vitprune.pipeline.benchmark.settings import VARIANTS
from vitprune.pipeline.cli import main

FILES = {
    "libs/__init__.py": "",
    "libs/core/__init__.py": """
        import yaml

        def load_config(path):
            cfg = yaml.safe_load(open(path))
            cfg.setdefault("model_name", "Fake")
            cfg.setdefault("model", {})
            return cfg
    """,
    # The evaluator reports, per tIoU, a tenth of the detections it was given plus a
    # hundredth of the annotated videos of the split: enough to see what reached it.
    "libs/utils/__init__.py": """
        import json

        import numpy as np

        def fix_random_seed(seed, include_cuda=True):
            pass

        class ANETdetection:
            def __init__(self, ant_file, split=None, tiou_thresholds=np.linspace(0.1, 0.5, 5)):
                database = json.load(open(ant_file))["database"]
                self.videos = [k for k, v in database.items() if v["subset"] == split]
                self.tiou_thresholds = tiou_thresholds

            def evaluate(self, preds, output_csv, verbose=True):
                found = np.full(len(self.tiou_thresholds), len(preds) / 10 + len(self.videos) / 100)
                return found, found.mean(), self.tiou_thresholds
    """,
    # One segment per video that reports what the model was given: the number of feature
    # steps, the duration, and the stride as the label.
    "libs/modeling/__init__.py": """
        import torch
        from torch import nn

        class Fake(nn.Module):
            def __init__(self, empty=False):
                super().__init__()
                self.layer = nn.Linear(4, 4)
                self.empty = empty

            def forward(self, video_list, is_training=True):
                assert not is_training
                out = []
                for item in video_list:
                    assert item["segments"] is None and item["labels"] is None
                    if self.empty:
                        segments, scores = torch.zeros(0, 2), torch.zeros(0)
                        labels = torch.zeros(0, dtype=torch.long)
                    else:
                        segments = torch.tensor(
                            [[float(item["feats"].shape[-1]), float(item["duration"])]])
                        scores = torch.tensor([0.5])
                        labels = torch.tensor([int(item["feat_stride"])])
                    out.append({"video_id": item["video_id"], "segments": segments,
                                "scores": scores, "labels": labels})
                return out

        def make_meta_arch(name, **kwargs):
            return Fake(**kwargs)
    """,
}
CONFIG = """
devices: ['cpu']
dataset: {{feat_stride: 99, num_frames: 16, default_fps: 30, input_dim: 6}}
model: {model}
"""


@dataclass
class FakeAma:
    out: Path
    project: Path
    config: Path
    weights: Path

    @property
    def args(self) -> dict:
        return {"out": self.out, "project": self.project, "config": self.config,
                "weights": self.weights, "python": sys.executable}

    def set_model_option(self, **options) -> None:
        self.config.write_text(CONFIG.format(model=json.dumps(options)), encoding="utf-8")


@pytest.fixture
def fake_ama(tmp_path) -> FakeAma:
    project = tmp_path / "AMA"
    for name, source in FILES.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    found = FakeAma(tmp_path / "out", project, tmp_path / "ama.yaml", tmp_path / "ama.pth")
    found.set_model_option()
    state = {"module.layer." + k: v for k, v in nn.Linear(4, 4).state_dict().items()}
    torch.save({"state_dict_ema": state}, found.weights)

    video = {"fps": 8.0, "frames": 64, "clips": 4}
    found.out.mkdir()
    (found.out / "run.json").write_text(json.dumps({
        "schedules": {"v": video}, "settings": {"stride": 16, "frame_step": 2},
    }), encoding="utf-8")
    for variant in VARIANTS:
        (found.out / "features" / variant).mkdir(parents=True)
        # 4 clips of 6 numbers: the AMA wants them as [6, 4], so the two cannot be confused.
        np.savez(found.out / "features" / variant / "v.npz", feats=np.zeros((4, 6), np.float32))
    return found


def test_the_ama_runs_every_variant_without_annotations(fake_ama):
    run_ama(**fake_ama.args)

    for variant in VARIANTS:
        folder = fake_ama.out / "ama" / variant
        rows = list(csv.reader((folder / "predictions.csv").open(encoding="utf-8")))
        assert rows[0] == ["video-id", "t-start", "t-end", "label", "score"]
        video, start, end, label, score = rows[1]
        assert (video, float(start), label) == ("v", 4.0, "16")  # T, and the run's stride, not 99
        assert float(end) == pytest.approx(64 / 30)  # frames over the AMA's default fps
        record = json.loads((folder / "ama-manifest.json").read_text(encoding="utf-8"))
        assert record["statistics"]["detections"] == 1
    assert (fake_ama.out / "ama" / "downstream.log").exists()


def test_a_variant_without_detections_gets_only_the_header(fake_ama):
    fake_ama.set_model_option(empty=True)

    run_ama(**fake_ama.args)

    text = (fake_ama.out / "ama" / "denso" / "predictions.csv").read_text(encoding="utf-8")
    assert text.splitlines() == ["video-id,t-start,t-end,label,score"]


def test_an_existing_output_is_refused_before_anything_runs(fake_ama):
    (fake_ama.out / "ama" / "vitprune").mkdir(parents=True)
    (fake_ama.out / "ama" / "vitprune" / "predictions.csv").write_text("x", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_ama(**fake_ama.args)

    assert not (fake_ama.out / "ama" / "denso").exists()
    assert not (fake_ama.out / "ama" / "job.json").exists()


def test_weights_that_do_not_match_the_model_write_nothing(fake_ama):
    torch.save({"state_dict_ema": {"other": torch.zeros(1)}}, fake_ama.weights)

    with pytest.raises(RuntimeError, match="exited with code"):
        run_ama(**fake_ama.args)

    assert not list((fake_ama.out / "ama").glob("*/predictions.csv"))


def test_a_variant_missing_its_features_is_named_before_anything_runs(fake_ama):
    (fake_ama.out / "features" / "vitprune" / "v.npz").unlink()

    with pytest.raises(FileNotFoundError, match="vitprune"):
        run_ama(**fake_ama.args)

    assert not (fake_ama.out / "ama" / "job.json").exists()


def test_the_inference_script_stays_independent_of_vitprune():
    tree = ast.parse(Path(ama.__file__).with_name("ama_infer.py").read_text(encoding="utf-8"))
    names = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(name.split(".")[0] == "vitprune" for name in names)


def test_the_ama_step_takes_everything_from_its_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        ama, "run_ama",
        lambda out, project, config, weights, python=None, labels=None: seen.update(
            out=out, project=project, config=config, weights=weights, python=python,
            labels=labels),
    )
    command = ["benchmark", "ama", "--out", "o", "--ama-project", "p", "--ama-config", "c.yaml",
               "--ama-weights", "a.pth", "--ama-python", "/x/python", "--labels", "l.json"]

    assert main(command) == 0

    assert seen == {"out": "o", "project": "p", "config": "c.yaml", "weights": "a.pth",
                    "python": "/x/python", "labels": "l.json"}


def test_the_manifest_records_the_features_and_the_interpreter_the_ama_ran_under(fake_ama):
    run_ama(**fake_ama.args)

    record = json.loads((fake_ama.out / "ama" / "denso" / "ama-manifest.json").read_text("utf-8"))
    features = fake_ama.out / "features" / "denso" / "v.npz"
    assert record["settings"]["feature_files"]["v"]["sha256"] == (
        hashlib.sha256(features.read_bytes()).hexdigest())
    assert record["environment"]["executable"] == sys.executable
    assert record["environment"]["torch"] == torch.__version__


def test_a_video_whose_fps_is_not_the_one_the_ama_reads_time_at_is_warned_about(fake_ama):
    lines = []

    run_ama(**fake_ama.args, progress=lines.append)  # the video is 8 fps, the config says 30

    warnings = [line for line in lines if line.startswith("warning:")]
    assert len(warnings) == 1 and "v" in warnings[0] and "30" in warnings[0]


def test_a_video_at_the_fps_the_ama_reads_time_at_gets_no_warning(fake_ama):
    run = json.loads((fake_ama.out / "run.json").read_text("utf-8"))
    run["schedules"]["v"]["fps"] = 30.0
    (fake_ama.out / "run.json").write_text(json.dumps(run), encoding="utf-8")
    lines = []

    run_ama(**fake_ama.args, progress=lines.append)

    assert not [line for line in lines if line.startswith("warning:")]


def _labels(path, stems=("v", "w")):
    database = {stem: {"subset": "training", "duration": 8.0, "annotations": [
        {"label_id": 3, "label": "x", "segment": [1.0, 2.0]}]} for stem in stems}
    path.write_text(json.dumps({"version": "1", "database": database}), encoding="utf-8")
    return path


def test_with_labels_the_ama_writes_the_map_of_each_variant(fake_ama, tmp_path):
    run_ama(**fake_ama.args, labels=_labels(tmp_path / "labels.json"))

    for variant in VARIANTS:
        found = json.loads((fake_ama.out / "ama" / variant / "map.json").read_text("utf-8"))
        assert found["tiou"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
        assert found["average"] == pytest.approx(0.11)  # one detection, one annotated video


def test_the_labels_keep_only_the_evaluated_videos(fake_ama, tmp_path):
    run_ama(**fake_ama.args, labels=_labels(tmp_path / "labels.json"))

    kept = json.loads((fake_ama.out / "ama" / "labels.json").read_text("utf-8"))["database"]
    assert set(kept) == {"v"} and kept["v"]["subset"] == "validation"


def test_a_variant_without_detections_has_a_map_of_zero(fake_ama, tmp_path):
    fake_ama.set_model_option(empty=True)

    run_ama(**fake_ama.args, labels=_labels(tmp_path / "labels.json"))

    found = json.loads((fake_ama.out / "ama" / "denso" / "map.json").read_text("utf-8"))
    assert found["map"] == [0.0] * 5 and found["average"] == 0.0


def test_an_evaluated_video_without_annotation_is_refused(fake_ama, tmp_path):
    with pytest.raises(ValueError, match="annotation"):
        run_ama(**fake_ama.args, labels=_labels(tmp_path / "labels.json", stems=("w",)))

    assert not (fake_ama.out / "ama" / "job.json").exists()
