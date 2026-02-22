#!/usr/bin/env bash
set -euo pipefail

PRED=${1:-outputs/report_predictions.jsonl}
python3 -m clincot.evaluation.runner --pred "${PRED}" --task report
