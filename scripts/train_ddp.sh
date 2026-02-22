#!/usr/bin/env bash
set -euo pipefail

NPROC=${1:-4}
CONFIG=${2:-configs/exp/clincot_base.yaml}
MODE=${3:-pref}
OUT=${4:-runs/${MODE}}

mkdir -p "${OUT}"

torchrun --nnodes=1 --nproc_per_node=${NPROC} \
  clincot/cli/train.py \
  --config "${CONFIG}" \
  --mode "${MODE}" \
  --output_dir "${OUT}"
