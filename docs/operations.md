# Pipeline operation

Install the local checkout with `pip install -e ".[pipeline,dev]"`. Base dependencies
are NumPy and Torch; `pipeline` adds OpenCV and PyYAML. Decord is used if separately
available. The old `torch` extra no longer exists.

## Commands

```bash
vitprune cost --retention 1.0 0.5 0.25 --save results/cost.json
vitprune check --weights w.pth --model-source /path/to/videomae
vitprune latency --weights w.pth --model-source /path/to/videomae --retention 1.0 0.5 0.25 --precision bf16 --save results/latency.json
vitprune drift --videos data/ --weights w.pth --model-source /path/to/videomae --retention 0.25 --clips 200 --save results/drift.json
vitprune run --video clip.mp4 --weights w.pth --model-source /path/to/videomae --out outputs/clip --compare
vitprune extract --videos data/ --weights w.pth --model-source /path/to/videomae --out features/r025 --retention 0.25 --stride 16 --frame-step 2
vitprune evaluate --features features/r025 --downstream-project /path/to/localizer --downstream-config config.yaml --downstream-weights best.pth --frame-step 2 --out runs/r025
```

`check`, `latency`, `drift`, `run` and `extract` support `--device cpu` and require
an external Giant backbone exposing `forward_features`. All verify dense equivalence
before application. The startup probe is outside benchmark timing.

`run --compare` is exploratory timing without warmup. Use `latency` for the formal
benchmark. Inspection in `run` uses the same autocast policy as feature extraction.
No command here recognises actions directly: `evaluate` invokes the external model.

## Files and provenance

- `run`: `VIDEO.npz`, `report.json`, `manifest.json`.
- `extract`: one `VIDEO.npz` per video and `manifest.json`.
- `evaluate`: `prepared-config.yaml`, `evaluation-manifest.json`, and the external
  model's results, requested at `predictions.csv`.
- Measurement `--save`: metrics plus arguments, environment and checkpoint identity
  where applicable.

Extraction refuses duplicate stems (including case-only duplicates) and existing
target features/manifests. Use distinct output folders per protocol. A failed run
can leave partial files; there is no automatic resume or transactional rollback.

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
