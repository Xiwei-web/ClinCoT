from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoImageProcessor, AutoTokenizer

from clincot.models.builder import ClinCoTConfig, build_model
from clincot.models.checkpoint_io import load_checkpoint
from clincot.models.language_backbone import LanguageConfig
from clincot.models.multimodal_projector import ProjectorConfig
from clincot.models.vision_backbone import VisionConfig
from clincot.data.preprocess import maybe_load_image


@dataclass
class InferenceBundle:
    model: torch.nn.Module
    tokenizer: Any
    image_processor: Any
    device: torch.device


def load_inference_bundle(config_path: str | Path, checkpoint_path: str | Path) -> InferenceBundle:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = ClinCoTConfig(
        vision=VisionConfig(**cfg["model"]["vision"]),
        language=LanguageConfig(**cfg["model"]["language"]),
        projector=ProjectorConfig(**cfg["model"]["projector"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.language.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_processor = AutoImageProcessor.from_pretrained(model_cfg.vision.model_name_or_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device).eval()
    load_checkpoint(model, Path(checkpoint_path), map_location="cpu", strict=False)

    return InferenceBundle(model=model, tokenizer=tokenizer, image_processor=image_processor, device=device)


def build_multimodal_inputs(bundle: InferenceBundle, image_path: str | Path, prompt: str) -> dict[str, torch.Tensor]:
    image = maybe_load_image(image_path)
    pixel_values = bundle.image_processor(images=image, return_tensors="pt")["pixel_values"].to(bundle.device)
    toks = bundle.tokenizer(prompt, return_tensors="pt")
    toks = {k: v.to(bundle.device) for k, v in toks.items()}
    return {"pixel_values": pixel_values, **toks}
