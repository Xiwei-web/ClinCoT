"""Method-level modules for ClinCoT."""

from .sdpo_loss import compute_dpo_loss, compute_sdpo_loss
from .clinical_scorer import ClinicalScoreAggregator, ScoreNormConfig
from .visual_cot import parse_bbox_from_text, crop_with_bbox, build_two_view_tensor
from .clincot_pipeline import ClinCoTPipeline, ClinCoTBatch
from .trainer_core import TrainerConfig, ClinCoTTrainer

__all__ = [
    "compute_dpo_loss",
    "compute_sdpo_loss",
    "ClinicalScoreAggregator",
    "ScoreNormConfig",
    "parse_bbox_from_text",
    "crop_with_bbox",
    "build_two_view_tensor",
    "ClinCoTPipeline",
    "ClinCoTBatch",
    "TrainerConfig",
    "ClinCoTTrainer",
]
