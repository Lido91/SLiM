#!/usr/bin/env bash
# SLiM self-supervised pre-training on sign-language pose (RTMPose, 69-joint Uni-Sign layout).
#
#   scripts/pretrain_sign.sh <run> <output_dir> [gpu_ids] [extra key=value ...]
#
# <run> is one of:  csl_daily  wlasl  openasl  asl    (slim/configs/pretrain/slim_sign_<run>.yaml)
#
# Pack the training poses first with tools/pack_sign_poses.py (see README, "Sign-language pose").
# The configs use 64 clips per GPU; the LR follows the sqrt scaling rule of the effective batch,
# so changing the GPU count changes the effective LR.
# Training resumes automatically from the last checkpoint in <output_dir>.
set -euo pipefail

RUN=${1:?usage: pretrain_sign.sh <csl_daily|wlasl|openasl|asl> <output_dir> [gpu_ids] [opts...]}
OUTDIR=${2:?missing output_dir}
GPUS=${3:-0,1,2,3}
shift 3 || shift 2
EXTRA=("$@")

CONFIG="slim/configs/pretrain/slim_sign_${RUN}.yaml"
if [[ ! -f "$CONFIG" ]]; then
  echo "unknown run: $RUN (no $CONFIG)" >&2; exit 1
fi

NPROC=$(awk -F',' '{print NF}' <<< "$GPUS")
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=1 KMP_AFFINITY=none MKL_THREADING_LAYER=GNU \
  torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29500}" \
  -m slim.train.train \
  --config-file "$CONFIG" \
  --output-dir "$OUTDIR" \
  ${EXTRA[@]+"${EXTRA[@]}"}
