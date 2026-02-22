#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/infer_vqa.sh config.yaml ckpt_dir input.jsonl output.jsonl
CONFIG=${1:-configs/exp/clincot_base.yaml}
CKPT=${2:-runs/pref/checkpoint-00000001}
INPUT=${3:-data/vqa_test.jsonl}
OUTPUT=${4:-outputs/vqa_predictions.jsonl}

python3 - <<PY
from clincot.utils.io import read_jsonl, write_jsonl
from clincot.inference.vqa_infer import VQAInferencer

samples = read_jsonl("${INPUT}")
runner = VQAInferencer("${CONFIG}", "${CKPT}")
outs = runner.infer_batch(samples)
write_jsonl("${OUTPUT}", outs)
print(f"saved {len(outs)} predictions -> ${OUTPUT}")
PY
