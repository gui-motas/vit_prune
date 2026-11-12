import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

BENCHMARK = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK))


def load_check():
    spec = importlib.util.spec_from_file_location("sep_check", BENCHMARK / "sep_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_features_match_clip_starts_and_reject_incomplete_schedule(tmp_path):
    check = load_check()
    folder = tmp_path / "features" / "prog_1536_1024_768_mascara"
    folder.mkdir(parents=True)
    (tmp_path / "run.json").write_text(json.dumps({
        "settings": {"stride": 16, "frame_step": 2},
        "schedules": {"v": {"frames": 80, "clips": 4}},
    }))
    features = np.array([[0, 10], [1, 11], [2, 12], [3, 13]], dtype=np.float32)
    np.savez(folder / "v.npz", feats=features)
    bench = SimpleNamespace(stride=16, frame_step=2, clip_starts=lambda frames: [0, 16, 32, 48])
    manifest = [{"video": "v", "starts": [48, 0]}, {"video": "v", "starts": [32]}]
    actual = check.load_baseline(bench, manifest, tmp_path)
    np.testing.assert_array_equal(actual[0], features[[3, 0]])
    np.testing.assert_array_equal(actual[1], features[[2]])
    np.savez(folder / "v.npz", feats=features[:-1])
    with pytest.raises(ValueError, match="v.*4.*3|v.*3.*4"):
        check.load_baseline(bench, manifest, tmp_path)
    np.savez(folder / "v.npz", feats=np.array(1.0))
    with pytest.raises(ValueError, match="v.*feats"):
        check.load_baseline(bench, manifest, tmp_path)


@pytest.mark.parametrize("initial,baseline,passed", [([1., 0.], [1., 0.], True),
    ([1., 0.], [1.02, 0.], False), ([.01, 0.], [0., .01], False)])
def test_control_gates_decision_against_saved_embeddings(tmp_path, monkeypatch, capsys,
                                                         initial, baseline, passed):
    import vitprune.bench.latency as latency
    import vitprune.core as core

    check = load_check()
    bench = SimpleNamespace(device="cpu")
    batches = [torch.tensor([initial]).repeat(8, 1) for _ in range(6)]
    manifest = [{"video": "v", "starts": list(range(8))} for _ in batches]
    monkeypatch.setattr(check, "setup", lambda args: (bench, None, torch.ones(256)))
    monkeypatch.setattr(check, "sample_batches", lambda bench: (batches, manifest))
    monkeypatch.setattr(check, "load_baseline", lambda *args: [
        np.tile(np.array(baseline, dtype=np.float32), (8, 1)) for _ in batches
    ], raising=False)
    monkeypatch.setattr(latency, "autocast", lambda *args: nullcontext())
    class Encoder:
        def __init__(self, model, geometry, plan):
            self.plan = plan

        def encode(self, clips):
            return clips

    monkeypatch.setattr(core, "PrunedEncoder", Encoder)
    monkeypatch.setattr(check, "diagnose", lambda encoder, clips, prior: (
        clips + (torch.tensor([0., .1]) if encoder.plan.separate_region else 0),
        {str(stage): {"cross": 0, "total": 1, "forced": 0} for stage in (2, 10, 20)}))
    monkeypatch.setattr(sys, "argv", ["sep_check.py", "--videos", "v", "--region", "r",
                                      "--weights", "w", "--model-source", "s", "--out",
                                      str(tmp_path)])
    status = check.main()
    result = json.loads((tmp_path / "check.json").read_text())
    output = capsys.readouterr().out
    assert result["control"]["passed"] is passed
    assert result["p_vs_p_sep"]["max_abs_delta"] == pytest.approx(.1)
    assert result["control"]["max_abs_delta_limit"] == .01
    assert result["control"]["min_cosine_limit"] == .9999
    if passed:
        assert status in (None, 0) and result["decision"] == "ADIAR" and "ADIAR" in output
    else:
        assert status == 2 and "decision" not in result
        assert "CONTROLE FALHOU" in output and "ADIAR" not in output and "PROSSEGUIR" not in output
        if initial == [.01, 0.]:
            assert result["control"]["max_abs_delta"] <= .01
            assert result["control"]["min_cosine"] < .9999


def test_incomplete_baseline_aborts_before_inference(tmp_path, monkeypatch, capsys):
    check = load_check()
    root = tmp_path / "b2"
    folder = root / "features" / "prog_1536_1024_768_mascara"
    folder.mkdir(parents=True)
    (root / "run.json").write_text(json.dumps({
        "settings": {"stride": 16, "frame_step": 2},
        "schedules": {"v": {"frames": 80, "clips": 4}},
    }))
    np.savez(folder / "v.npz", feats=np.ones((3, 2), dtype=np.float32))
    bench = SimpleNamespace(device="cpu", stride=16, frame_step=2,
                            clip_starts=lambda frames: [0, 16, 32, 48])
    batches = [torch.ones(8, 2) for _ in range(6)]
    manifest = [{"video": "v", "starts": [0] * 8} for _ in batches]
    monkeypatch.setattr(check, "B2", root)
    monkeypatch.setattr(check, "setup", lambda args: (bench, None, None))
    monkeypatch.setattr(check, "sample_batches", lambda bench: (batches, manifest))
    monkeypatch.setattr(sys, "argv", ["sep_check.py", "--videos", "v", "--region", "r",
                                      "--weights", "w", "--model-source", "s", "--out",
                                      str(tmp_path / "out")])
    assert check.main() == 2
    result = json.loads((tmp_path / "out" / "check.json").read_text())
    assert not result["control"]["passed"] and "3 saved rows" in result["control"]["error"]
    assert "decision" not in result
    output = capsys.readouterr().out
    assert "CONTROLE FALHOU" in output and "ADIAR" not in output and "PROSSEGUIR" not in output
