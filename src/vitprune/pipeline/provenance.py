"""Recording the conditions under which each result was produced.

A number without provenance is an anecdote. When measurement happens on a different
machine from development, the chance of comparing two runs that only look alike is
high: a different library version, a parameter changed without being noted, a
checkpoint that was replaced.

Each extraction writes, next to its results, a record of what was in force: the
parameters, the checkpoint identified by the SHA-256 of its content rather than by
name, the library versions and the machine.

That is what allows answering, months later, whether two numbers are comparable
without having to guess.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class SourceFile:
    """A file identified by its content rather than its name."""

    path: str
    bytes: int
    modified: str
    sha256: str = ""

    @classmethod
    def of(cls, path: str | Path) -> SourceFile:
        info = Path(path).stat()
        stamp = datetime.fromtimestamp(info.st_mtime, tz=timezone.utc)
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(str(Path(path).resolve()), info.st_size, stamp.isoformat(), digest.hexdigest())


@dataclass
class Manifest:
    """Everything needed to judge whether two results are comparable."""

    settings: dict[str, Any]
    weights: SourceFile | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    created: str = ""
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.created:
            self.created = datetime.now(tz=timezone.utc).isoformat()
        if not self.environment:
            self.environment = describe_environment()

    def write(self, folder: str | Path, name: str = MANIFEST_NAME) -> Path:
        destination = Path(folder) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return destination


def read_manifest(folder: str | Path) -> dict[str, Any]:
    """Read the record of a previous run.

    Used to verify, before evaluating, that the settings recorded at extraction match
    the settings of the evaluation. See :mod:`vitprune.pipeline.evaluation`.
    """
    source = Path(folder) / MANIFEST_NAME
    if not source.exists():
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} in {folder}. Without it there is no way to know which "
            f"settings produced these embeddings, and comparing results becomes a guess."
        )
    return json.loads(source.read_text(encoding="utf-8"))


def describe_environment() -> dict[str, Any]:
    """Versions and machine, so an environment difference does not go unnoticed."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    for package in ("vitprune", "numpy", "PyYAML", "opencv-python"):
        try:
            info[package] = version(package)
        except PackageNotFoundError:
            info[package] = None
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    info["vitprune_source_sha256"] = digest.hexdigest()
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        info["torch"] = None
    return info
