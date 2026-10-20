# Historical implementation and evidence

The Obsidian notes originally described revision
`f53f3ec5c68193300304923d71e1c87b1ded97cd` under `C:/Python/ViTPrune`.
The current `C:/Python/vit_prune` checkout has no `.git` metadata, so that revision
cannot be asserted as the origin of the present files.

## Earlier application workflow

The notes refer to `scripts/comparar_inferencia.py`, `scripts/visualizar_selecao.py`,
an AMA-specific in-process localizer, an Excalidraw diagram and temporary outputs.
Those artifacts are not present in this checkout. Their commands are not supported
entry points of the current library. No files were removed to make this distinction.

Current equivalents are partial, not identical: `PrunedEncoder.select()` exposes
selection; `run --compare` produces features and an exploratory comparison; `evaluate`
invokes an external temporal project. There is no built-in segment-comparison runner.

## Reported CPU experiment

Earlier documentation reports 35 clips from a 19-second video, dense encoder time
1141.3 s, pruned time 285.5 s, measured speedup 4.00x and predicted 3.93x. It also
reports cosine mean 0.81, worst 0.24, and a `Drinking` segment near 9.25–16.64 s
with confidence 0.824.

These numbers are retained as historical reports, not re-executed or independently
verified measurements. Original weights, outputs and complete environment records
are not included. They do not establish dataset quality or GPU performance.

See [experiments.md](experiments.md) for the current validation protocol and
[operations.md](operations.md) for executable commands.
