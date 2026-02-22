from __future__ import annotations

from typing import Any

from .utils.text import normalize_text, safe_div


def _contains_all_facts(text: str, facts: list[str]) -> float:
    if not facts:
        return 0.0
    text_n = normalize_text(text)
    hit = 0
    for fact in facts:
        if normalize_text(fact) in text_n:
            hit += 1
    return safe_div(hit, len(facts))


def compute_factuality_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Simple factuality proxy.

    Expected optional fields:
    - key_facts: list[str]
    - hallucination_terms: list[str]
    """
    total = len(rows)
    fact_cov = 0.0
    hallucination_rate = 0.0

    for row in rows:
        pred = row.get("pred", "")
        key_facts = row.get("key_facts", [])
        hall_terms = row.get("hallucination_terms", [])

        fact_cov += _contains_all_facts(pred, key_facts)

        pred_n = normalize_text(pred)
        has_hall = 0
        for t in hall_terms:
            if normalize_text(t) in pred_n:
                has_hall = 1
                break
        hallucination_rate += has_hall

    return {
        "num_samples": float(total),
        "fact_coverage": safe_div(fact_cov, total),
        "hallucination_rate": safe_div(hallucination_rate, total),
    }
