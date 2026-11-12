"""What defines a benchmark run, and the record the later steps read back.

The run is defined by its command line alone. Its inputs are written to ``run.json`` next
to the results, so that ``vitprune benchmark ama`` and ``vitprune benchmark report`` need
only the results folder. The policy study includes fixed, masked and progressive variants;
progressive variants use a region-guided cut followed by global merges.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from vitprune.core.encoder import PruningPlan
from vitprune.core.geometry import ClipGeometry, validate_prune_layer
from vitprune.core.policy import FixedBudget
from vitprune.core.priors import top_rows_prior
from vitprune.pipeline.video import _clip_starts, clip_span, find_videos

GEOMETRY = ClipGeometry.videomae_giant()

#: The dense VideoMAE, and ViT-Prune with no mask, with the fixed mask and with the smart
#: mask, the driver region. The pruned variants share one policy, so they keep the same
#: tokens per clip; the masks only change which ones.
VARIANTS = ("denso", "vitprune", "vitprune_mascara_fixa", "vitprune_mascara_inteligente")

#: Names the policy study gives its own variants, never a policy file passed by name.
RESERVED = ("denso", "legacy_original")

#: Written by the run, read by the AMA step and the report.
RUN_RECORD = "run.json"


@dataclass
class Benchmark:
    """Everything that defines one run of the variants over the evaluated videos.

    Two studies share the run. The mask study compares the dense VideoMAE with ViT-Prune
    without a mask, with the fixed mask and with the driver region, under ``policy``. The
    policy study, chosen by giving ``policies``, ``fixed``, ``proportional``, ``masked`` or
    ``progressive``, compares budgets instead: the dense VideoMAE, ``policy`` when given (as
    ``legacy_original``), each named policy file, and fixed budgets, plain, with size-biased
    attention, selecting inside the driver region, or progressively merging after a
    region-guided cut. The last two use a mask.

    Args:
        videos: folder of the videos under evaluation; never the calibration videos.
        out: the results folder.
        policy, region: written by ``vitprune calibrate`` and ``vitprune region``. The mask
            study needs the policy; the policy study may go without it.
        weights, model_source: the backbone checkpoint and the VideoMAE folder.
        evaluate: stems of the videos to evaluate; every video in ``videos`` when empty.
        device, precision, batch: how the backbone runs; ``precision`` is ``off`` (fp32,
            as the AI City feature extraction ran), ``fp16`` or ``bf16``.
        stride, frame_step: how clips are sampled; the extraction values.
        prune_layer: the block after which tokens are pruned.
        fixed_rows, fixed_weight: the fixed mask lowers the score of this many rows from
            the top of the grid by this factor.
        show: clips per video kept for the figures, spread evenly over it.
        learnable_positions: the checkpoint carries a learned positional embedding.
        policies: ``(name, file)`` of the policies to compare, legacy or quality-aware.
        fixed: tokens of each fixed-budget variant, ``fixo_<tokens>``.
        proportional: tokens of each fixed-budget variant with size-biased attention,
            ``fixo_<tokens>_pa``.
        masked: tokens of each fixed-budget variant whose selection takes the driver
            ``region`` as its spatial prior, ``fixo_<tokens>_mascara``.
        attention: backbone attention implementation, ``model`` or ``sdpa``.
        progressive: cut and later merges as ``(block, tokens)`` pairs.
        progressive_separate: also compare region-separated progressive merging.
        allow_region_overlap: allow and record a region calibrated on evaluated footage.
    """

    videos: Path
    out: Path
    policy: Path | None
    region: Path
    weights: Path
    model_source: Path
    evaluate: tuple[str, ...] = ()
    device: str = "cuda"
    precision: str = "off"
    batch: int = 8
    stride: int = 16
    frame_step: int = 2
    prune_layer: int = 2
    fixed_rows: int = 4
    fixed_weight: float = 0.15
    show: int = 6
    learnable_positions: bool = False
    policies: tuple[tuple[str, Path], ...] = ()
    fixed: tuple[int, ...] = ()
    proportional: tuple[int, ...] = ()
    masked: tuple[int, ...] = ()
    attention: str = "model"
    progressive: tuple[tuple[int, int], ...] = ()
    allow_region_overlap: bool = False
    progressive_separate: bool = False

    def __post_init__(self) -> None:
        for name in ("videos", "out", "region", "weights", "model_source"):
            setattr(self, name, Path(getattr(self, name)))
        self.policy = None if self.policy is None else Path(self.policy)
        self.evaluate = tuple(self.evaluate)
        self.policies = tuple((str(name), Path(path)) for name, path in self.policies)
        self.fixed, self.proportional = tuple(self.fixed), tuple(self.proportional)
        self.masked = tuple(self.masked)
        if self.attention not in ("model", "sdpa"):
            raise ValueError("attention must be 'model' or 'sdpa'")
        if not isinstance(self.progressive_separate, bool):
            raise ValueError("progressive_separate must be a bool")
        if self.progressive_separate and not self.progressive:
            raise ValueError("--progressive-separate requires --progressive")
        if self.progressive:
            if (not isinstance(self.progressive, (tuple, list))
                    or any(not isinstance(stage, (tuple, list)) or len(stage) != 2
                           for stage in self.progressive)):
                raise ValueError("progressive must be (block, tokens) pairs")
            self.progressive = tuple(tuple(stage) for stage in self.progressive)
            if len(self.progressive) < 2 or self.progressive[0][0] != self.prune_layer:
                raise ValueError("progressive needs the cut and at least one later stage")
            PruningPlan(prune_layer=self.prune_layer, budget=FixedBudget(self.progressive[0][1]),
                        stages=self.progressive[1:]).validate_for(GEOMETRY)
        names = [name for name, _ in self.policies]
        for name in names:
            generated = name.startswith(("fixo_", "prog_"))
            if not re.fullmatch(r"[a-z0-9_]+", name) or name in RESERVED or generated:
                raise ValueError(f"a policy variant needs a name of a-z, 0-9 and _, not one of "
                                 f"{RESERVED} nor fixo_* or prog_*: got {name!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"policy variant names must be unique, got {names}")
        for levels in (self.fixed, self.proportional, self.masked):
            if len(set(levels)) != len(levels):
                raise ValueError(f"fixed budgets must be unique, got {levels}")
            for tokens in levels:
                FixedBudget(tokens).validate_for(GEOMETRY)
        if self.policy is None and not self.policy_study:
            raise ValueError("the mask study needs --policy, written by 'vitprune calibrate'")
        if self.precision not in ("off", "fp16", "bf16"):
            raise ValueError("precision must be 'off', 'fp16' or 'bf16'")
        if self.batch < 1 or self.stride < 1 or self.frame_step < 1 or self.show < 1:
            raise ValueError("batch, stride, frame_step and show must be at least 1")
        validate_prune_layer(self.prune_layer, GEOMETRY.depth)
        top_rows_prior(GEOMETRY.grid_size, self.fixed_rows, self.fixed_weight)  # validates

    @property
    def policy_study(self) -> bool:
        """Budgets compared with one another, rather than masks under one policy."""
        return bool(self.policies or self.fixed or self.proportional or self.masked
                    or self.progressive)

    @property
    def progressive_name(self) -> str:
        return "prog_" + "_".join(str(tokens) for _, tokens in self.progressive) + "_mascara"

    @property
    def variants(self) -> tuple[str, ...]:
        if not self.policy_study:
            return VARIANTS
        return ("denso", *(("legacy_original",) if self.policy is not None else ()),
                *(name for name, _ in self.policies), *(f"fixo_{k}" for k in self.fixed),
                *(f"fixo_{k}_pa" for k in self.proportional),
                *(f"fixo_{k}_mascara" for k in self.masked),
                *((self.progressive_name,) if self.progressive else ()),
                *((self.progressive_name + "_sep",) if self.progressive_separate else ()))

    @property
    def span(self) -> int:
        return clip_span(GEOMETRY, self.frame_step)

    @property
    def fixed_mask(self) -> dict:
        return {"rows": self.fixed_rows, "weight": self.fixed_weight}

    @property
    def record(self) -> Path:
        return self.out / RUN_RECORD

    @property
    def clips(self) -> Path:
        return self.out / "clips.jsonl"

    @property
    def features(self) -> Path:
        return self.out / "features"

    @property
    def selections(self) -> Path:
        return self.out / "selection"

    @property
    def frames(self) -> Path:
        return frames_folder(self.out)

    def clip_starts(self, total_frames: int) -> list[int]:
        """First frame of every clip, as the extraction pipeline samples them."""
        return _clip_starts(total_frames, self.span, self.stride)

    def evaluated(self) -> list[Path]:
        """The videos under evaluation: the ones named in ``evaluate``, or every video."""
        found = find_videos(self.videos)
        if not self.evaluate:
            if not found:
                raise FileNotFoundError(f"no video found in {self.videos}")
            return found
        by_stem = {path.stem: path for path in found}
        missing = set(self.evaluate) - set(by_stem)
        if missing:
            raise FileNotFoundError(
                f"evaluation videos not found in {self.videos}: {sorted(missing)}"
            )
        return [by_stem[stem] for stem in sorted(set(self.evaluate))]


def frames_folder(out: str | Path) -> Path:
    """The middle frame of each shown clip, kept for the figures."""
    return Path(out) / "frames"


@dataclass
class RunRecord:
    """A run as ``run.json`` describes it."""

    out: Path
    run: dict

    @classmethod
    def read(cls, out: str | Path) -> RunRecord:
        out = Path(out)
        path = out / RUN_RECORD
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found: run 'vitprune benchmark run' with --out {out} first"
            )
        run = json.loads(path.read_text(encoding="utf-8"))
        videos = run.get("schedules")
        if not isinstance(videos, dict) or not all(
            isinstance(v, dict) and {"fps", "frames", "clips"} <= set(v) for v in videos.values()
        ):
            raise ValueError(
                f"{path} was written by an older vitprune; run again with "
                "'vitprune benchmark run'"
            )
        return cls(out, run)

    @property
    def stems(self) -> list[str]:
        """The evaluated videos."""
        return list(self.run["schedules"])

    @property
    def variants(self) -> list[str]:
        """The variants the run compared; the dense one first."""
        return self.settings.get("variants", list(VARIANTS))

    def video(self, stem: str) -> dict:
        """``fps``, ``frames`` and ``clips`` of an evaluated video."""
        return self.run["schedules"][stem]

    @property
    def settings(self) -> dict:
        return self.run["settings"]

    @property
    def fixed_mask(self) -> dict:
        return self.settings["fixed_mask"]

    @property
    def prune_layer(self) -> int:
        return self.settings["prune_layer"]

    @property
    def frames(self) -> Path:
        return frames_folder(self.out)

    @property
    def figures(self) -> Path:
        return self.out / "figuras"
