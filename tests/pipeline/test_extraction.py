"""Exercise extraction through its public interface, without actual video or weights."""

import hashlib
import json

import numpy as np
import pytest
import torch

from vitprune.core import PrunedEncoder, PruningPlan
from vitprune.core.testing import reference_backbone, reference_geometry
from vitprune.pipeline import extraction
from vitprune.pipeline import video as reader
from vitprune.pipeline.provenance import SourceFile


@pytest.fixture
def encoder(uniform_budget):
    geometry = reference_geometry(depth=2)
    plan = PruningPlan(
        spatial_prior=torch.linspace(0.1, 1.0, geometry.tokens_per_slice),
        budget=uniform_budget(geometry, 16),
    )
    return PrunedEncoder(reference_backbone(geometry), geometry, plan)


def test_extraction_records_the_actual_protocol_and_fingerprints(tmp_path, monkeypatch, encoder):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"fake checkpoint")
    definitions = tmp_path / "source" / "models" / "modeling_feature.py"
    definitions.parent.mkdir(parents=True)
    definitions.write_text("# external model", encoding="utf-8")
    frames = np.zeros((32, 32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(reader, "read_frames", lambda path, start=0: iter(frames[start:]))
    folder = tmp_path / "features"
    path = extraction.extract_folder(
        encoder, video, folder, batch=2, device="cpu", precision="off",
        weights_path=weights, model_source=definitions.parents[1], progress=lambda _: None,
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema_version"] == 2
    assert record["settings"]["spatial_prior"] == encoder.plan.spatial_prior.tolist()
    assert record["settings"]["geometry"]["heads"] == encoder.geometry.heads
    assert record["settings"]["geometry"]["mlp_ratio"] == encoder.geometry.mlp_ratio
    assert record["settings"]["effective_precision"] == "off"
    assert record["weights"]["sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert record["settings"]["model_definition"]["sha256"]
    assert record["environment"]["vitprune_source_sha256"]
    assert record["settings"]["budget"]["version"] == encoder.plan.budget.version
    assert record["outputs"]["clip"]["budget"]["tokens"] == [16, 16]
    assert record["cost"]["tokens_pruned"] == 16
    with np.load(folder / "clip.npz") as result:
        assert result["feats"].shape == (2, encoder.geometry.embed_dim)
        assert result["feats"].dtype == np.float32
        assert np.isfinite(result["feats"]).all()


def test_duplicate_stems_fail_before_encoding_or_creating_outputs(tmp_path, encoder):
    for name in ("a/clip.mp4", "b/clip.avi"):
        video = tmp_path / name
        video.parent.mkdir()
        video.write_bytes(b"")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="unique"):
        extraction.extract_folder(encoder, tmp_path, out)
    assert not out.exists()


def test_existing_features_are_not_silently_overwritten(tmp_path, encoder):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    out = tmp_path / "features"
    out.mkdir()
    existing = out / "clip.npz"
    existing.write_bytes(b"user data")
    with pytest.raises(FileExistsError):
        extraction.extract_folder(encoder, video, out)
    assert existing.read_bytes() == b"user data"


@pytest.mark.parametrize("batch", [0, -1, True])
def test_encode_clips_rejects_invalid_batches(encoder, batch):
    with pytest.raises(ValueError, match="batch"):
        extraction.encode_clips(encoder, torch.randn(*encoder.geometry.input_shape()), batch=batch)


def test_encode_clips_rejects_empty_batches(encoder):
    with pytest.raises(ValueError, match="at least one clip"):
        extraction.encode_clips(encoder, torch.empty(*encoder.geometry.input_shape(batch=0)))


def test_non_finite_features_are_not_saved(monkeypatch):
    geometry = reference_geometry(depth=2)
    encoder = PrunedEncoder(reference_backbone(geometry), geometry, PruningPlan.dense())
    monkeypatch.setattr(encoder, "encode", lambda clips: torch.full((1, 32), float("nan")))
    with pytest.raises(ValueError, match="non-finite"):
        extraction.encode_clips(
            encoder, torch.randn(*encoder.geometry.input_shape()), device="cpu", precision="off"
        )


def test_source_identity_uses_contents_not_only_size_and_name(tmp_path):
    path = tmp_path / "checkpoint"
    path.write_bytes(b"aa")
    first = SourceFile.of(path)
    path.write_bytes(b"bb")
    second = SourceFile.of(path)
    assert first.bytes == second.bytes
    assert first.sha256 != second.sha256
