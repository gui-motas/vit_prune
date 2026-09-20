"""Token grid arithmetic for a video clip.

No neural network runs here. This module answers counting questions only: how many
tokens a clip produces, how they are laid out in space and time, and how many remain
after pruning.

All of that arithmetic lives in one place on purpose. When ``image_size /
patch_size`` and ``frames_per_clip / tubelet_size`` are recomputed across several
modules, a configuration mistake surfaces as a tensor shape error dozens of lines
later, with nothing pointing back at the cause.
"""

from __future__ import annotations

from dataclasses import dataclass

# VideoMAE v2 Giant (ViT-g/14).
GIANT_IMAGE_SIZE = 224
GIANT_PATCH_SIZE = 14
GIANT_TUBELET_SIZE = 2
GIANT_FRAMES_PER_CLIP = 16
GIANT_EMBED_DIM = 1408
GIANT_DEPTH = 40
GIANT_HEADS = 16
GIANT_MLP_RATIO = 48 / 11

# VideoMAE v2 Large (ViT-L/16).
LARGE_PATCH_SIZE = 16
LARGE_EMBED_DIM = 1024
LARGE_DEPTH = 24
LARGE_HEADS = 16
LARGE_MLP_RATIO = 4.0


@dataclass(frozen=True)
class ClipGeometry:
    """How many visual tokens a clip produces, and how they are arranged.

    The backbone receives a clip of ``frames_per_clip`` frames. Each frame is resized
    to ``image_size`` and split into patches of side ``patch_size``. Neighbouring
    frames are grouped into tubelets of ``tubelet_size``, which shortens the temporal
    axis.

    For the Giant defaults that gives 16 by 16 patches in each of 8 temporal slices,
    so 2048 tokens per clip.

    These fields describe a checkpoint, not a preference. Changing them without
    changing the model makes weight loading fail its shape validation.

    Prefer the named constructors over the raw one: they carry the values that go
    together, so a partial mismatch cannot be introduced by accident.
    """

    image_size: int = GIANT_IMAGE_SIZE
    patch_size: int = GIANT_PATCH_SIZE
    tubelet_size: int = GIANT_TUBELET_SIZE
    frames_per_clip: int = GIANT_FRAMES_PER_CLIP
    embed_dim: int = GIANT_EMBED_DIM
    depth: int = GIANT_DEPTH
    heads: int = GIANT_HEADS
    mlp_ratio: float = GIANT_MLP_RATIO

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        if self.frames_per_clip % self.tubelet_size:
            raise ValueError(
                f"frames_per_clip ({self.frames_per_clip}) must be divisible by "
                f"tubelet_size ({self.tubelet_size})"
            )
        if self.embed_dim % self.heads:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by heads ({self.heads})"
            )

    # -- named constructors -------------------------------------------------
    @classmethod
    def videomae_giant(cls, frames_per_clip: int = GIANT_FRAMES_PER_CLIP) -> ClipGeometry:
        """VideoMAE v2 Giant: patch 14, 1408 dimensions, 40 blocks."""
        return cls(
            image_size=GIANT_IMAGE_SIZE,
            patch_size=GIANT_PATCH_SIZE,
            tubelet_size=GIANT_TUBELET_SIZE,
            frames_per_clip=frames_per_clip,
            embed_dim=GIANT_EMBED_DIM,
            depth=GIANT_DEPTH,
            heads=GIANT_HEADS,
            mlp_ratio=GIANT_MLP_RATIO,
        )

    @classmethod
    def videomae_large(cls, frames_per_clip: int = GIANT_FRAMES_PER_CLIP) -> ClipGeometry:
        """VideoMAE v2 Large: patch 16, 1024 dimensions, 24 blocks."""
        return cls(
            image_size=GIANT_IMAGE_SIZE,
            patch_size=LARGE_PATCH_SIZE,
            tubelet_size=GIANT_TUBELET_SIZE,
            frames_per_clip=frames_per_clip,
            embed_dim=LARGE_EMBED_DIM,
            depth=LARGE_DEPTH,
            heads=LARGE_HEADS,
            mlp_ratio=LARGE_MLP_RATIO,
        )

    @classmethod
    def image_vit(
        cls,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
    ) -> ClipGeometry:
        """A plain image transformer, modelled as a clip with one temporal slice.

        Pruning works, but the motion term is always zero: with a single slice there
        is no previous one to compare against. Set ``motion_weight=0.0`` in the
        pruning plan so the score reflects that instead of silently relying on a
        constant.
        """
        return cls(
            image_size=image_size,
            patch_size=patch_size,
            tubelet_size=1,
            frames_per_clip=1,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
        )

    # -- derived counts -----------------------------------------------------
    @property
    def grid_size(self) -> int:
        """Patches per side of a frame. 224 / 14 = 16."""
        return self.image_size // self.patch_size

    @property
    def tokens_per_slice(self) -> int:
        """Tokens in a single temporal slice. 16 x 16 = 256."""
        return self.grid_size**2

    @property
    def temporal_slices(self) -> int:
        """Temporal slices per clip. 16 frames / tubelet 2 = 8."""
        return self.frames_per_clip // self.tubelet_size

    @property
    def num_tokens(self) -> int:
        """Tokens in a whole clip. 8 x 256 = 2048."""
        return self.temporal_slices * self.tokens_per_slice

    @property
    def head_dim(self) -> int:
        """Width seen by each attention head. 1408 / 16 = 88."""
        return self.embed_dim // self.heads

    @property
    def has_temporal_axis(self) -> bool:
        """False for a single-slice geometry, where the motion term cannot apply."""
        return self.temporal_slices > 1

    def kept_per_slice(self, retention: float) -> int:
        """Tokens kept in each temporal slice after pruning.

        ``retention`` is the retention ratio. The result is rounded and never drops
        below 1, so a slice cannot end up empty.

        >>> ClipGeometry.videomae_giant().kept_per_slice(0.25)
        64
        """
        if not 0.0 < retention <= 1.0:
            raise ValueError(f"retention must be in (0, 1], got {retention}")
        return max(1, round(self.tokens_per_slice * retention))

    def kept_tokens(self, retention: float) -> int:
        """Tokens left in the whole clip after pruning."""
        return self.temporal_slices * self.kept_per_slice(retention)

    def input_shape(self, batch: int = 1) -> tuple[int, int, int, int, int]:
        """Shape the backbone expects: batch, channels, frames, height, width."""
        return (batch, 3, self.frames_per_clip, self.image_size, self.image_size)

    def describe(self) -> str:
        """One-line summary, for report headers."""
        return (
            f"patch {self.patch_size}px, embed_dim {self.embed_dim}, {self.depth} blocks | "
            f"{self.num_tokens} tokens ({self.temporal_slices} slices x {self.tokens_per_slice})"
        )
