#!/usr/bin/env bash
# The quality-aware budget policy (QABP) experiment, stage by stage, on the Jetson AGX Orin.
#
#   bash benchmark/run_qabp.sh p0           # P0: the budget curve on calibration clips, gates G1/G2
#   bash benchmark/run_qabp.sh colecao      # the records of the 13 calibration videos
#   bash benchmark/run_qabp.sh legacy       # the legacy policy recalibrated on the fit videos
#   bash benchmark/run_qabp.sh calibracao   # the QABP and its ablation without the probe; gate G3
#   bash benchmark/run_qabp.sh benchmark    # only after G3 passes: the variants, the AMA, the report
#
# Every stage resumes after an interruption. Results go to benchmark/results/qabp; nothing
# under benchmark/results is overwritten. The split is fixed: fit and holdout by driver,
# the evaluated videos and 38159_NoAudio_5 (same driver as an evaluated one) never calibrate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUNBUFFERED=1

PY=.venv/bin/python
CLI=("$PY" -m vitprune.pipeline.cli)
M=/home/gsilv257/DAR-sys/models-weights
WEIGHTS=$M/videomaev2/videomae-v2_inference.pth
SOURCE=$M/AIcity2024-track3/feature_extraction
MODEL=(--weights "$WEIGHTS" --model-source "$SOURCE" --device cuda)
RUN=(--precision off --batch 8 --stride 16 --frame-step 2 --prune-layer 2)
OUT=benchmark/results/qabp
FIT=(13522_NoAudio_7 28557_NoAudio_5 28557_NoAudio_7 30932_NoAudio_5 30932_NoAudio_7
     31903_NoAudio_5 31903_NoAudio_7 33682_NoAudio_5 33682_NoAudio_7)
HOLDOUT=(35418_NoAudio_5 35418_NoAudio_7 36305_NoAudio_3 36305_NoAudio_5)

videos() {
    mkdir -p "$OUT/videos/ajuste" "$OUT/videos/holdout"
    for stem in "${FIT[@]}"; do
        ln -sfn "$PWD/benchmark/videos/Dashboard_user_id_$stem.MP4" "$OUT/videos/ajuste/"
    done
    for stem in "${HOLDOUT[@]}"; do
        ln -sfn "$PWD/benchmark/videos/Dashboard_user_id_$stem.MP4" "$OUT/videos/holdout/"
    done
}

p0() {
    videos
    "${CLI[@]}" budget-curve "${MODEL[@]}" "${RUN[@]}" --videos "$OUT/videos/ajuste" \
        --clips-per-video 3 --depths 2 3 6 10 20 30 40 --out "$OUT/p0/records"
    [ -f "$OUT/p0/p0.json" ] || "$PY" benchmark/qabp_p0.py --weights "$WEIGHTS" \
        --model-source "$SOURCE" --records "$OUT/p0/records" --videos "$OUT/videos/ajuste" \
        --out "$OUT/p0"
}

for stage in "$@"; do
    date
    "$stage"
done
date
