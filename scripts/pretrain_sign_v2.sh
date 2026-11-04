#!/usr/bin/env bash
# Fresh V2 architecture; existing V1 runs/checkpoints remain separate.
# Usage: bash scripts/pretrain_sign_v2.sh <output_dir> [gpu_ids] [extra training options...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=${1:?usage: pretrain_sign_v2.sh <output_dir> [gpu_ids] [opts...]}
GPUS=${2:-0,1,2,3}
if (( $# >= 2 )); then shift 2; else shift; fi
exec bash scripts/pretrain_sign.sh v2_openasl "$OUT" "$GPUS" "$@"
