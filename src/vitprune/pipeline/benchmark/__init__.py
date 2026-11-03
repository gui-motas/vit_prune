"""The experiment: the four variants on a set of videos, the AMA on their features, and
the report, with the dense pipeline as the reference.

Three steps, each a ``vitprune benchmark`` command, each reading what the one before
wrote under ``--out``:

* :mod:`~vitprune.pipeline.benchmark.inference` -- every clip of the evaluated videos
  through the dense VideoMAE and three ViT-Prune variants (no mask, the fixed mask, the
  driver region);
* :mod:`~vitprune.pipeline.benchmark.ama` -- the AMA temporal localizer on each
  variant's features, run directly: no annotation is needed;
* :mod:`~vitprune.pipeline.benchmark.report` -- a spreadsheet and figures, comparing each
  variant with the dense run.

The policy and the region come from ``vitprune calibrate`` and ``vitprune region`` run
on other videos of the same camera, so nothing is judged on the footage it was fitted
to. The report needs the ``benchmark`` extra:

    pip install "vitprune[benchmark]"
"""
