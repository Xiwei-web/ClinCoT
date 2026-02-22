from __future__ import annotations

import re
from collections import Counter


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize_words(s: str) -> list[str]:
    s = normalize_text(s)
    return re.findall(r"[a-z0-9]+", s)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def f1_score(pred: str, gt: str) -> float:
    p = tokenize_words(pred)
    g = tokenize_words(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    cp = Counter(p)
    cg = Counter(g)
    overlap = sum((cp & cg).values())
    precision = safe_div(overlap, len(p))
    recall = safe_div(overlap, len(g))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
