#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/data_build_pairs.sh input_candidates.jsonl output_pairs.jsonl 0.2

INPUT=${1:-data/candidates.jsonl}
OUTPUT=${2:-data/train_pref.jsonl}
MIN_GAP=${3:-0.0}

python3 - <<PY
from clincot.utils.io import read_jsonl, write_jsonl
from clincot.data.pair_builder import build_pairs_from_scored_candidates

rows = read_jsonl("${INPUT}")
pairs = build_pairs_from_scored_candidates(rows, min_score_gap=float("${MIN_GAP}"))
write_jsonl("${OUTPUT}", pairs)
print(f"built pairs: {len(pairs)} -> ${OUTPUT}")
PY
