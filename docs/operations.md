# Pipeline operation

Install the local checkout with `pip install -e ".[pipeline,dev]"`. Base dependencies
are NumPy and Torch; `pipeline` adds OpenCV, which decodes every video, and PyYAML.
The old `torch` extra no longer exists.

## Commands

```bash
vitprune cost --policy policy.json --save results/cost.json
vitprune check --weights w.pth --model-source /path/to/videomae
vitprune calibrate --videos calibration/ --weights w.pth --model-source /path/to/videomae --levels 256 512 768 --out policy.json
vitprune latency --weights w.pth --model-source /path/to/videomae --policy policy.json --videos data/ --precision bf16 --save results/latency.json
vitprune drift --videos data/ --weights w.pth --model-source /path/to/videomae --policy policy.json --clips 200 --save results/drift.json
vitprune run --video clip.mp4 --weights w.pth --model-source /path/to/videomae --policy policy.json --out outputs/clip --compare
vitprune extract --videos data/ --weights w.pth --model-source /path/to/videomae --policy policy.json --out features/adaptive --stride 16 --frame-step 2
vitprune evaluate --features features/adaptive --downstream-project /path/to/localizer --downstream-config config.yaml --downstream-weights best.pth --frame-step 2 --out runs/adaptive
```

`calibrate` measures redundancy and motion on calibration videos, running only the
blocks up to the cut, and writes the adaptive budget to a policy file.
`--clips-per-video N` takes N clips spread evenly over each video instead of every
clip; the first clips of a video are not a sample of it. Calibration
videos must not be part of the evaluation. A policy file is never overwritten, and a
policy is refused when used with a different `--prune-layer` or `--frame-step` than it
was calibrated with. Use the same stride and precision as the extraction.

`--policy` selects the adaptive budget in `cost`, `latency`, `drift`, `run` and
`extract`. Without it, `run` and `extract` produce the dense baseline; `latency` and
`drift` require it. `cost` without a policy takes the levels from `--tokens` (default
256 512 768). `latency` times the dense path and the policy on `--batch` real clips,
since the budget depends on the content; `drift` compares them on `--clips` clips.
Both spread their clips evenly over every video in `--videos`.

`check`, `latency`, `drift`, `run` and `extract` support `--device cpu` and require
an external Giant backbone exposing `forward_features`. All verify dense equivalence
before application. The startup probe is outside benchmark timing.

`run --compare` is exploratory timing without warmup. Use `latency` for the formal
benchmark. Inspection in `run` uses the same autocast policy as feature extraction.
No command here recognises actions directly: `evaluate` invokes the external model.

## Files and provenance

- `calibrate`: the policy file, with the policy, its version and the calibration
  videos, clip count, cut, sampling and checkpoint identity.
- `run`: `VIDEO.npz`, `report.json`, `manifest.json`.
- `extract`: one `VIDEO.npz` per video and `manifest.json`. With a policy, the
  manifest records the policy and, per video, each clip's tokens, regime, redundancy
  and motion; `feats` is unchanged.
- `evaluate`: `prepared-config.yaml`, `evaluation-manifest.json`, and the external
  model's results, requested at `predictions.csv`.
- Measurement `--save`: metrics plus arguments, environment and checkpoint identity
  where applicable.

Extraction refuses duplicate stems (including case-only duplicates) and existing
target features/manifests. Use distinct output folders per protocol. An interrupted
run keeps `manifest.partial.json`: the same command run again skips the videos it
lists and carries on, and refuses if the settings or the checkpoint changed. The
finished `manifest.json` appears only when every video is done.

Manifest schema 2 records full geometry, actual prior values, requested and effective
autocast precision, backbone class, package-source fingerprint, package versions,
and SHA-256 identities for weights, source videos and feature files. When supplied,
`model_source` identifies `models/modeling_feature.py`, not every dependency of that
external project. Hashing large inputs adds I/O outside the timed encoder benchmark.

Evaluation checks recorded stride and feature width when available. An explicit
`--frame-step` (or `dataset.frame_step`) is checked against recorded sampling and
fails when the record is missing. Other temporal fields, FPS and label conventions
remain integration responsibilities. It records the original/prepared config,
weights, entry point, labels, command, extraction manifest and process exit status.
This adapter never trains, but cannot police arbitrary code in an external script.

Old manifests are readable. Missing prior values, hashes or temporal metadata are
not invented; rerun extraction when stronger provenance is required.
