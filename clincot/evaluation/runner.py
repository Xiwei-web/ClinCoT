from __future__ import annotations

import argparse
import json

from .utils.io import read_jsonl
from .vqa_metrics import compute_vqa_metrics
from .report_metrics import compute_report_metrics
from .factuality_metrics import compute_factuality_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=str, required=True)
    p.add_argument("--task", type=str, choices=["vqa", "report", "factuality", "all"], default="all")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.pred)

    outputs = {}
    if args.task in {"vqa", "all"}:
        outputs["vqa"] = compute_vqa_metrics(rows)
    if args.task in {"report", "all"}:
        outputs["report"] = compute_report_metrics(rows)
    if args.task in {"factuality", "all"}:
        outputs["factuality"] = compute_factuality_metrics(rows)

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
