# Pipeline operation

Install the local checkout with `pip install -e ".[pipeline,dev]"`. Base dependencies
are NumPy and Torch; `pipeline` adds OpenCV, which decodes every video, and PyYAML;
`region` adds Ultralytics for the person detector; `benchmark` adds pandas, openpyxl and
matplotlib for the report. The old `torch` extra no longer exists.

## Commands

```bash
vitprune cost --policy policy.json --save results/cost.json
vitprune check --weights w.pth --model-source /path/to/videomae
vitprune calibrate --videos calibration/ --weights w.pth --model-source /path/to/videomae --levels 256 512 768 --out policy.json
vitprune region --videos calibration/ --detector yolo26s.pt --out region.json
vitprune budget-curve --videos calibration/ --weights w.pth --model-source /path/to/videomae --levels 512 1024 1536 2048 --clips-per-video 48 --precision off --out records/
vitprune calibrate-quality --records records/ --holdout VIDEO_A VIDEO_B --tolerance 0.1 --out quality_policy.json
vitprune latency --weights w.pth --model-source /path/to/videomae --policy policy.json --videos data/ --precision bf16 --save results/latency.json
vitprune drift --videos data/ --weights w.pth --model-source /path/to/videomae --policy policy.json --clips 200 --save results/drift.json
vitprune run --video clip.mp4 --weights w.pth --model-source /path/to/videomae --policy policy.json --out outputs/clip --compare
vitprune extract --videos data/ --weights w.pth --model-source /path/to/videomae --policy policy.json --out features/adaptive --stride 16 --frame-step 2
vitprune evaluate --features features/adaptive --downstream-project /path/to/localizer --downstream-config config.yaml --downstream-weights best.pth --frame-step 2 --out runs/adaptive
vitprune benchmark run --videos eval/ --policy policy.json --region region.json --weights w.pth --model-source /path/to/videomae --out outputs/benchmark
vitprune benchmark run --videos eval/ --region region.json --policies qabp=quality_policy.json --fixed 1536 1024 --proportional 1536 --weights w.pth --model-source /path/to/videomae --out outputs/policies
vitprune benchmark ama --out outputs/benchmark --ama-project /path/to/AMA --ama-config config.yaml --ama-weights best.pth
vitprune benchmark ama --out outputs/policies --ama-project /path/to/AMA --ama-config config.yaml --ama-weights best.pth --labels labels.json
vitprune benchmark report --out outputs/benchmark
```

`calibrate` measures redundancy and motion on calibration videos, running only the
blocks up to the cut, and writes the adaptive budget to a policy file.
`--clips-per-video N` takes N clips spread evenly over each video instead of every
clip; the first clips of a video are not a sample of it. Calibration
videos must not be part of the evaluation. A policy file is never overwritten, and a
policy is refused when used with a different `--prune-layer`, `--frame-step` or
checkpoint than it was calibrated with. Use the same stride and precision as the
extraction.

`--policy` selects the adaptive budget in `cost`, `latency`, `drift`, `run` and
`extract`. Without it, `run` and `extract` produce the dense baseline; `latency` and
`drift` require it. `cost` without a policy takes the levels from `--tokens` (default
256 512 768). `latency` times the dense path and the policy on `--batch` real clips,
since the budget depends on the content; `drift` compares them on `--clips` clips.
Both spread their clips evenly over every video in `--videos`.

`region` runs a COCO person detector on the middle frame of every calibration clip, with
the rule of the AI City preprocessing: each video's crop is its largest confident person
box, and the region is the median of the crops. `--detector` takes the weights of any
detector Ultralytics loads, such as
`yolo26s.pt`; it needs the `region` extra, not the VideoMAE source. `--device` takes
`cuda`, `cuda:N` or `cpu`; `--clips-per-video` works as in `calibrate`.
`--region region.json` then applies it in `drift`, `run` and `extract`; it requires
`--policy` and excludes `--prior-top-rows`. A region file is never overwritten and is
refused for a backbone with another grid.

`budget-curve` measures, on clips spread over each video (`--clips-per-video`), how far
each budget moves the final embedding from the dense one: the blocks up to the cut run
once, then the remaining blocks once per level in `--levels`, the dense path always among
them. Each clip's record holds its features at the cut (motion, redundancy and the probe of
compressibility), and per level `1 - cos`, the relative L2 distance, the norms, whether a
classification head gives the dense class, the merge's error per token and, after each
block in `--depths`, the drift of the mean of the tokens. It is offline work, for
calibration and diagnosis: inference never runs a clip twice. `--size-biased-attention`
measures with proportional attention. The records folder resumes video by video, a video
counting as done once `sources.json` lists it, and is refused under other settings. The driver of a video is the number after `user_id_` in its
name, or the name itself.

`calibrate-quality` fits the quality-aware budget on those records, without a GPU: the
videos in `--holdout` are kept out of the fit, and no driver may be on both sides. The
policy reads a clip's motion, redundancy and probe error at the cut, places it in one of
`--bins` bins, and keeps the fewest tokens whose calibrated risk (the 90th percentile of
`1 - cos` in that bin on the fit clips) is within `--tolerance`; otherwise, and whenever
it cannot trust itself (a value outside the calibrated range, not finite, or a bin under
`--min-support` clips or `--min-videos` videos), the dense path. `--no-probe` writes the
same table without the probe, the ablation. The file holds the policy, its calibration
(checkpoint, cut, sampling, precision, videos and drivers of each side) and its
validation on the holdout, with the verdict of the registered criteria. The risk is an
empirical estimate, not a guarantee.

A quality-aware policy file goes wherever `--policy` goes. It is refused under another
precision, cut, frame step or checkpoint, and with another selection than the one it was
calibrated with: another `--motion-weight`, a spatial prior or `--size-biased-attention`.

`benchmark` is the experiment in three steps, each reading what the one before wrote
under `--out` (default `outputs/benchmark`). It needs no annotation: the dense pipeline is
the reference, and each variant is compared with what the dense gave.

- `benchmark run` puts every clip of the evaluated videos through the dense backbone and
  three ViT-Prune variants that share one policy: no mask, a fixed mask over the top
  rows, and the driver region. `--videos` holds the evaluated videos, and `--evaluate
  STEM ...` picks some of them, by name without extension. `--policy` and `--region` come
  from `calibrate` and `region`, run on other videos. The policy is refused when
  `--prune-layer`, `--frame-step` or the checkpoint differ from the ones it was calibrated
  with; use the same `--stride` and `--precision` too, which are not checked.
  `--fixed-rows` and `--fixed-weight` shape the fixed mask, and `--show N` keeps N clips
  per video, spread over it, for the figures. A video's rows are written before its features
  and each feature file whole, so an interrupted run resumes with the same command: a video
  without all its features is redone, its leftover rows dropped first. A folder holding
  another policy, region or setting is refused.
- `benchmark run --policies NAME=PATH ... --fixed TOKENS ... --proportional TOKENS ...
  --masked TOKENS ... --progressive K1 B:K2 ...` runs the policy study instead: the dense backbone and budgets. Each
  policy file, legacy or quality-aware, becomes the variant `NAME`; each fixed budget the
  variant `fixo_TOKENS`, with size-biased attention `fixo_TOKENS_pa`, and selecting only
  inside the `--region` `fixo_TOKENS_mascara`. `--progressive` makes the variant
  `prog_<K1>_<K2>_..._mascara`: after the region-guided initial cut, later ToMe-style
  merges match globally across temporal slices. `--policy`,
  optional here, adds `legacy_original`. The order the variants are timed in turns batch
  by batch. A quality-aware policy calibrated on an evaluated video or driver is refused;
  a legacy one is only reported, in `run.json`. A `--region` calibrated on them is refused
  too whenever it chooses tokens: with `--masked`, `--progressive`, and in the mask study.
  `--allow-region-overlap` overrides only the region guard for exploratory runs; the
  opt-in and overlap are recorded. `--attention {model,sdpa}` selects the backbone's own
  attention (default) or SDPA for every variant.
- `benchmark ama` runs the AMA temporal localizer, the pipeline's next stage, on each
  variant's features. It does not use the AMA's `eval.py`, which compares detections with
  annotations: the model runs from `ama_infer.py`, a script started in the AMA's folder
  under `--ama-python`, the interpreter of the AMA's environment (default: the one
  running vitprune; vitprune need not be installed in the other). A pure-Python
  `nms_1d_cpu` stands in for the AMA's compiled extension. The weights must match the
  model exactly. A variant already run is refused, before anything starts. With
  `--labels`, annotations in the AMA's JSON layout covering every evaluated video, the
  AMA's own evaluator also scores each variant against them: `map.json`, the mAP per tIoU
  from 0.1 to 0.5 and its average.
- `benchmark report` writes `benchmark-vitprune.xlsx` and the figures in `figuras/`. It
  works without `benchmark ama`, only without the detection columns. Per variant it gives
  the share of clips where the head's class is the dense one's; and, for the detections of
  the dense run from `--min-score` up, the share each variant finds again with the same
  class and a tIoU of `--tiou` or more (recall), the share of its own detections the
  dense run confirms (precision) and the mean best tIoU. These measure how much pruning
  changes the dense result, not whether that result is right: where the dense is wrong, a
  variant that is wrong the same way agrees. With the annotations, it also gives the mAP,
  the annotated actions each variant finds and the share of its detections they confirm,
  its false positives, the error of its boundaries and the confidence of its detections,
  and the head's class against each annotated clip and action. Boundaries and confidence
  are measured on the actions the variant and the dense run both find, with the dense
  run's own on those same actions beside them: over each run's own finds, a variant that
  loses the hard actions would look closer than the dense run. The cosine and relative L2 distance to the dense
  embedding, and the measured time per clip, sit beside them as auxiliary measures.

`benchmark ama` and `benchmark report` take the evaluated videos and the sampling from
the `run.json` the first step wrote.

`check`, `latency`, `drift`, `run` and `extract` support `--device cpu` and require
an external Giant backbone exposing `forward_features`. All verify dense equivalence
before application. The startup probe is outside benchmark timing.

`run --compare` is exploratory timing without warmup. Use `latency` for the formal
benchmark. Inspection in `run` uses the same autocast policy as feature extraction.
No command here recognises actions directly: `evaluate` invokes the external model.

## Files and provenance

- `calibrate`: the policy file, with the policy, its version and the calibration
  videos, clip count, cut, sampling and checkpoint identity.
- `budget-curve`: under `--out`, `settings.json`, `sources.json` (each video's file,
  driver and clip count), one `VIDEO.jsonl` per video and `summary.json`.
- `calibrate-quality`: the policy file, never overwritten: `policy` (with
  `policy_type: quality_aware_v1` and its version), `calibration` and `validation`.
- `region`: the region file, with the grid, the region, its box and, per calibration
  video, how many clips found the driver and the crop, plus the detection rule and the
  detector weights.
- `benchmark run`: under `--out`, `run.json` (inputs, checksums, settings and the clips of
  each video), `clips.jsonl` (one row per clip and variant), `features/VARIANT/` in the
  `extract` format with its manifest, and the tokens kept and frames the figures use.
  For `prog_*`, the `tokens` column records the initial cut count, not the later counts.
  `benchmark ama` adds `ama/`: per variant `predictions.csv` (`video-id`, `t-start`,
  `t-end`, `label`, `score`, in seconds) and `ama-manifest.json`, plus `job.json` and
  `downstream.log`; `benchmark report` the workbook and `figuras/`.
- `run`: `VIDEO.npz`, `report.json`, `manifest.json`.
- `extract`: one `VIDEO.npz` per video and `manifest.json`. With a policy, the
  manifest records the policy and, per video, each clip's tokens, regime, redundancy
  and motion; `feats` is unchanged.
- `evaluate`: `prepared-config.yaml`, `evaluation-manifest.json`, `downstream.log` with
  everything the external model printed, and its results, requested at
  `predictions.csv`. A folder that already holds an evaluation is refused.
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
