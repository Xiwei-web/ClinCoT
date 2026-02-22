from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=str, required=True, help="prediction jsonl with fields: id,pred,gt")
    return p.parse_args()


def normalize(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def main() -> None:
    args = parse_args()

    total = 0
    hit = 0
    with open(Path(args.pred), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            if normalize(row.get("pred", "")) == normalize(row.get("gt", "")):
                hit += 1

    acc = hit / max(total, 1)
    print(json.dumps({"total": total, "exact_match": acc}, indent=2))


if __name__ == "__main__":
    main()
