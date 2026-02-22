from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence
import numpy as np


@dataclass
class ScoreNormConfig:
    clip_min: float = 0.75
    clip_max: float = 1.25
    eps: float = 1e-6


class ClinicalScoreAggregator:
    """Utility for score aggregation and normalization.

    Typical inputs:
    - textual clinical relevance scores from one/multiple evaluators
    - visual confidence score from lesion localization tools
    """

    def __init__(self, norm_cfg: ScoreNormConfig | None = None) -> None:
        self.norm_cfg = norm_cfg or ScoreNormConfig()

    @staticmethod
    def aggregate_text_scores(scores: Sequence[float], reducer: str = "mean") -> float:
        if len(scores) == 0:
            return 0.0
        if reducer == "mean":
            return float(mean(scores))
        if reducer == "max":
            return float(max(scores))
        if reducer == "min":
            return float(min(scores))
        raise ValueError(f"Unsupported reducer={reducer}")

    @staticmethod
    def fuse_multisource(
        text_score: float,
        visual_score: float | None = None,
        alpha_text: float = 0.7,
    ) -> float:
        if visual_score is None:
            return float(text_score)
        alpha_text = float(alpha_text)
        return alpha_text * float(text_score) + (1.0 - alpha_text) * float(visual_score)

    def normalize_scores(self, scores: Iterable[float]) -> list[float]:
        vals = np.asarray(list(scores), dtype=np.float32)
        if vals.size == 0:
            return []
        mu = float(vals.mean())
        std = float(vals.std())
        std = max(std, self.norm_cfg.eps)

        z = (vals - mu) / std
        clipped = np.clip(z, self.norm_cfg.clip_min, self.norm_cfg.clip_max)
        return clipped.tolist()
