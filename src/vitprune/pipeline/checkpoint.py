"""Reading a checkpoint and installing it, refusing what does not fit.

A wrong checkpoint rarely fails cleanly. It usually loads almost correctly, leaving a
few layers at their random initialisation, and the problem only shows up later as a
poor result with no explanation -- which is indistinguishable from a method that did
not work.

So loading here is suspicious by nature: it verifies the critical shapes **before**
installing anything, and reports what was missing or left over. Failing early with a
useful message costs minutes; failing late with none costs days.

A training checkpoint often carries the optimiser state alongside the weights, which
is why a file can be several times larger than the model. This module knows how to
find the weights inside that wrapper.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vitprune.core.geometry import ClipGeometry

#: Keys a checkpoint may wrap the weights under, in order of preference.
_WRAPPERS = ("module", "model", "state_dict")

#: Prefixes some checkpoints prepend to every layer name.
_PREFIXES = ("module.", "backbone.", "encoder.")

#: Shapes that must match exactly, or the load would be silently wrong.
_CRITICAL = ("patch_embed.proj.weight", "pos_embed")


@dataclass(frozen=True)
class LoadReport:
    """What happened while installing the weights."""

    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.missing and not self.unexpected

    def describe(self) -> str:
        if self.clean:
            return "weights installed with nothing missing or left over"
        parts = []
        if self.missing:
            parts.append(f"{len(self.missing)} missing (e.g. {self.missing[0]})")
        if self.unexpected:
            parts.append(f"{len(self.unexpected)} unused (e.g. {self.unexpected[0]})")
        return "weights installed with " + " and ".join(parts)


def read_weights(path: str | Path) -> dict[str, torch.Tensor]:
    """Extract the weights from a checkpoint, unwrapping as needed.

    Handles files that store the weights under a key and layer names that carry a
    prefix. The file is memory-mapped rather than read: a training checkpoint can carry
    an optimiser state several times the size of the model, and only the weights are
    ever touched.
    """
    try:
        raw: Any = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:
        # Only the zip format can be mapped; files in the legacy format are read whole.
        raw = torch.load(str(path), map_location="cpu", weights_only=False)

    for wrapper in _WRAPPERS:
        if isinstance(raw, dict) and wrapper in raw and hasattr(raw[wrapper], "keys"):
            raw = raw[wrapper]
            break

    if not hasattr(raw, "keys"):
        raise ValueError(
            f"{path} does not contain a recognisable weight dictionary (looked under {_WRAPPERS})"
        )
    return {_strip(name): value for name, value in raw.items()}


def install_weights(model: nn.Module, weights: dict[str, torch.Tensor]) -> LoadReport:
    """Verify the critical shapes, then install the weights.

    Raises before installing anything if a critical shape disagrees, because a
    partial load produces a model that looks functional and is silently wrong.
    """
    expected = model.state_dict()
    for name in _CRITICAL:
        if name in weights and name in expected:
            got, want = tuple(weights[name].shape), tuple(expected[name].shape)
            if got != want:
                raise ValueError(
                    f"shape mismatch at {name}: the file has {got}, the model expects "
                    f"{want}. This usually means the patch size, the tubelet size or "
                    f"the kind of positional embedding differs from the model that "
                    f"produced the file."
                )

    missing, unexpected = model.load_state_dict(weights, strict=False)
    return LoadReport(tuple(missing), tuple(unexpected))


def load_videomae(
    weights_path: str | Path,
    source_dir: str | Path,
    geometry: ClipGeometry | None = None,
    device: str = "cpu",
    builder_name: str = "vit_giant_patch14_224",
    learnable_positions: bool = False,
) -> nn.Module:
    """Build a VideoMAE backbone and install the weights into it.

    The model definition is not distributed with this package: it belongs to the
    VideoMAE project. ``source_dir`` points at the folder containing
    ``models/modeling_feature.py``.

    Keeping that definition outside is deliberate. This package observes the
    backbone; it does not modify it. Copying the model code in would create two
    versions that can diverge without warning.

    Args:
        weights_path: the checkpoint file.
        source_dir: folder holding the VideoMAE model definition.
        geometry: expected clip geometry, used only for reporting.
        device: where to place the model.
        builder_name: the factory function to call, for variants other than Giant.
        learnable_positions: build with a learned positional embedding instead of a
            fixed sinusoidal one.
    """
    geometry = geometry or ClipGeometry.videomae_giant()
    builder = _import_builder(source_dir, builder_name)
    model = builder(use_learnable_pos_emb=learnable_positions)

    report = install_weights(model, read_weights(weights_path))
    if report.missing:
        raise ValueError(
            f"{report.describe()}. Missing layers keep their random initialisation, "
            f"which produces poor results with no error message. "
            f"First missing: {report.missing[:5]}"
        )
    return model.eval().to(device)


def _import_builder(source_dir: str | Path, builder_name: str):
    """Import the factory that builds the model, from the given folder."""
    source = Path(source_dir).resolve()
    if not (source / "models" / "modeling_feature.py").exists():
        raise FileNotFoundError(
            f"models/modeling_feature.py not found in {source}. Point --model-source "
            f"at the VideoMAE folder that holds the model definition."
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    import models.modeling_feature as definitions

    actual_source = getattr(definitions, "__file__", None)
    expected_source = source / "models" / "modeling_feature.py"
    if actual_source is None or Path(actual_source).resolve() != expected_source:
        raise ValueError(
            "models.modeling_feature was imported from a different source. "
            "Use a fresh Python process with the intended --model-source."
        )
    if not hasattr(definitions, builder_name):
        raise AttributeError(f"{builder_name} not found in models/modeling_feature.py")
    return getattr(definitions, builder_name)


def disable_activation_checkpointing(model: nn.Module) -> nn.Module:
    """Turn off activation recomputation, which only makes sense while training.

    It trades compute for memory by running each block twice. During inference that
    is pure waste, and it would distort any timing measurement.
    """
    if hasattr(model, "with_cp"):
        model.with_cp = False
    for block in getattr(model, "blocks", []):
        if hasattr(block, "with_cp"):
            block.with_cp = False
    return model


def _strip(name: str) -> str:
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name
