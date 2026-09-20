"""Motion-guided visual token pruning for VideoMAE-style video transformers.

The method lives in :mod:`vitprune.core` and depends only on numpy and torch.
Measurement lives in :mod:`vitprune.bench`. Video I/O, checkpoint loading and the
evaluation driver live in :mod:`vitprune.pipeline`, behind the ``pipeline`` extra.

Start with ``docs/integration.md`` to add it to an existing project.
"""

from vitprune.core import ClipGeometry, PrunedEncoder, PruningPlan, check_backbone

__version__ = "0.1.0"

__all__ = ["ClipGeometry", "PrunedEncoder", "PruningPlan", "check_backbone", "__version__"]
