from __future__ import annotations

from typing import Any

from .utils.text import f1_score, normalize_text, safe_div


def compute_vqa_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute simple VQA metrics.

    Expected row fields:
    - pred
    - gt
    - optional: answer_type in {"open", "closed"}
    """
    total = len(rows)
    exact_hits = 0
    f1_sum = 0.0

    open_total = 0
    open_hits = 0
    closed_total = 0
    closed_hits = 0

    for row in rows:
        pred = normalize_text(row.get("pred", ""))
        gt = normalize_text(row.get("gt", ""))

        exact = int(pred == gt)
        exact_hits += exact
        f1_sum += f1_score(pred, gt)

        atype = row.get("answer_type", None)
        if atype == "open":
            open_total += 1
            open_hits += exact
        elif atype == "closed":
            closed_total += 1
            closed_hits += exact

    return {
        "num_samples": float(total),
        "exact_match": safe_div(exact_hits, total),
        "f1": safe_div(f1_sum, total),
        "open_accuracy": safe_div(open_hits, open_total),
        "closed_accuracy": safe_div(closed_hits, closed_total),
    }
