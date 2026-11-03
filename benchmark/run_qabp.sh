#!/usr/bin/env bash
# The quality-aware budget policy (QABP) experiment, stage by stage, on the Jetson AGX Orin.
#
#   bash benchmark/run_qabp.sh p0           # P0: the budget curve on calibration clips, gates G1/G2
#   bash benchmark/run_qabp.sh atencao      # the P0 panel again, with proportional attention
#   bash benchmark/run_qabp.sh sanidade     # dense and 1536 through the AMA on one selection video
#   bash benchmark/run_qabp.sh selecao      # the downstream validation on the selection videos
#   bash benchmark/run_qabp.sh colecao      # the records of the calibration videos
#   bash benchmark/run_qabp.sh legacy       # the legacy policy recalibrated on the fit videos
#   bash benchmark/run_qabp.sh calibracao   # the QABP and its ablation without the probe; gate G3
#   bash benchmark/run_qabp.sh benchmark    # only after G3 passes: the variants, the AMA, the report
#
# Every stage resumes after an interruption. Results go to benchmark/results/qabp; nothing
# under benchmark/results is overwritten. The split is fixed, by driver (revision 3): the
# selection set chooses configurations, the fit set fits the policy, the holdout validates
# it; the evaluated videos and 38159_NoAudio_5 (same driver as an evaluated one) never
# calibrate nor select. All sixteen local videos are in the AMA's training subset.
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
AMA=$M/AIcity2024-track3/AMA
AMA_RUN=(--ama-project "$AMA" --ama-config "$AMA/configs/aicity_ego_vitl.yaml"
         --ama-weights "$M/AMA/ckpt/aicity_ego_vitl_ckpt/best_performance.pth.tar"
         --labels "$M/AIcity2024-track3/data/label_A1-train_A1-val.json")
REGION=region-novo.json
OUT=benchmark/results/qabp
# The P0 panel as it ran: the fit set before revision 3 took the selection set out of it.
P0=(13522_NoAudio_7 28557_NoAudio_5 28557_NoAudio_7 30932_NoAudio_5 30932_NoAudio_7
    31903_NoAudio_5 31903_NoAudio_7 33682_NoAudio_5 33682_NoAudio_7)
SELECTION=(28557_NoAudio_5 28557_NoAudio_7 30932_NoAudio_5 30932_NoAudio_7)
FIT=(13522_NoAudio_7 31903_NoAudio_5 31903_NoAudio_7 33682_NoAudio_5 33682_NoAudio_7)
HOLDOUT=(35418_NoAudio_5 35418_NoAudio_7 36305_NoAudio_3 36305_NoAudio_5)

link() {  # link FOLDER STEM...: a folder holding links to exactly these videos
    local folder="$OUT/videos/$1"
    shift
    mkdir -p "$folder"
    rm -f "$folder"/*.MP4
    for stem in "$@"; do
        ln -s "$PWD/benchmark/videos/Dashboard_user_id_$stem.MP4" "$folder/"
    done
}

videos() {
    link p0 "${P0[@]}"
    link selecao "${SELECTION[@]}"
    link sanidade "${SELECTION[0]}"
    link ajuste "${FIT[@]}"
    link holdout "${HOLDOUT[@]}"
}

ama() {  # ama RUN: the AMA and its mAP against the annotations, then the report
    [ -f "$OUT/$1/ama/denso/map.json" ] || "${CLI[@]}" benchmark ama --out "$OUT/$1" "${AMA_RUN[@]}"
    "${CLI[@]}" benchmark report --out "$OUT/$1"
}

p0() {
    videos
    "${CLI[@]}" budget-curve "${MODEL[@]}" "${RUN[@]}" --videos "$OUT/videos/p0" \
        --clips-per-video 3 --depths 2 3 6 10 20 30 40 --out "$OUT/p0/records"
    [ -f "$OUT/p0/p0.json" ] || "$PY" benchmark/qabp_p0.py --weights "$WEIGHTS" \
        --model-source "$SOURCE" --records "$OUT/p0/records" --videos "$OUT/videos/p0" \
        --out "$OUT/p0"
}

atencao() {
    videos
    "${CLI[@]}" budget-curve "${MODEL[@]}" "${RUN[@]}" --videos "$OUT/videos/p0" \
        --clips-per-video 3 --levels 1024 1536 2048 --depths 2 3 6 10 20 30 40 \
        --size-biased-attention --out "$OUT/p0-pa/records"
}

sanidade() {
    videos
    "${CLI[@]}" benchmark run "${MODEL[@]}" "${RUN[@]}" --videos "$OUT/videos/sanidade" \
        --region "$REGION" --fixed 1536 --out "$OUT/selecao-sanidade"
    ama selecao-sanidade
}

selecao() {
    videos
    "${CLI[@]}" benchmark run "${MODEL[@]}" "${RUN[@]}" --videos "$OUT/videos/selecao" \
        --region "$REGION" --fixed 1536 1024 512 --proportional 1536 1024 --out "$OUT/selecao"
    ama selecao
}

for stage in "$@"; do
    date
    "$stage"
done
date
