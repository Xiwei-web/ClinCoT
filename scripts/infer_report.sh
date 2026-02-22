#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/exp/clincot_base.yaml}
CKPT=${2:-runs/pref/checkpoint-00000001}
INPUT=${3:-data/report_test.jsonl}
OUTPUT=${4:-outputs/report_predictions.jsonl}

python3 - <<PY
from clincot.utils.io import read_jsonl, write_jsonl
from clincot.inference.report_infer import ReportInferencer

samples = read_jsonl("${INPUT}")
runner = ReportInferencer("${CONFIG}", "${CKPT}")
outs = runner.infer_batch(samples)
write_jsonl("${OUTPUT}", outs)
print(f"saved {len(outs)} predictions -> ${OUTPUT}")
PY
