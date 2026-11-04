#!/usr/bin/env bash
# Linear evaluation (Table 1): frozen encoder + linear classifier grid.
#
#   scripts/eval_linear.sh <checkpoint> <output_dir> <ntu60|ntu120|wlasl> <xsub|xview|-> [gpu_ids]
#
# `xview` selects X-View on NTU-60 and X-Set on NTU-120.
#
# Set CLASSIFIER=<path> to skip training and evaluate an already-trained classifier instead:
#   CLASSIFIER=checkpoints/linear/ntu60_xsub.pth scripts/eval_linear.sh ...
set -euo pipefail

CKPT=${1:?usage: eval_linear.sh <checkpoint> <output_dir> <ntu60|ntu120|wlasl> <xsub|xview|-> [gpu_ids]}
OUTDIR=${2:?missing output_dir}
DATASET=${3:?missing dataset}
BENCH=${4:?missing benchmark}
GPUS=${5:-0,1,2,3}

DATA_ROOT=${DATA_ROOT:-./data}
CONFIG=slim/configs/eval/slim_eval.yaml
DATASET_ARGS=()
case "$DATASET" in
  ntu60)  DATA_PATH="$DATA_ROOT/ntu";    NUM_CLASSES=60 ;;
  ntu120) DATA_PATH="$DATA_ROOT/ntu120"; NUM_CLASSES=120 ;;
  # sign-pose packs from tools/pack_sign_poses.py; <benchmark> is ignored, the classifier is picked on dev and reported on test
  wlasl)  DATA_PATH="$DATA_ROOT/sign/WLASL"; NUM_CLASSES=2000; BENCH=xsub
          CONFIG=slim/configs/eval/slim_sign_eval.yaml
          DATASET_ARGS=(--val-dataset SignPose:split=VAL --test-datasets SignPose:split=TEST) ;;
  *) echo "unknown dataset: $DATASET (expected ntu60, ntu120 or wlasl)" >&2; exit 1 ;;
esac

NPROC=$(awk -F',' '{print NF}' <<< "$GPUS")
mkdir -p "$OUTDIR"

EXTRA=()
if [[ -n "${CLASSIFIER:-}" ]]; then
  EXTRA=(--eval-only --classifier-fpath "$CLASSIFIER")
fi

CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=1 KMP_AFFINITY=none MKL_THREADING_LAYER=GNU \
  torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29501}" \
  -m slim.eval.linear \
  --config-file "$CONFIG" \
  --pretrained-weights "$CKPT" \
  --output-dir "$OUTDIR" \
  --data-path "$DATA_PATH" \
  --num-classes "$NUM_CLASSES" \
  --benchmark "$BENCH" \
  ${DATASET_ARGS[@]+"${DATASET_ARGS[@]}"} \
  ${EXTRA[@]+"${EXTRA[@]}"}
