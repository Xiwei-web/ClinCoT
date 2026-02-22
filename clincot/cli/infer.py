from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import torch
from transformers import AutoImageProcessor, AutoTokenizer

from clincot.models.builder import ClinCoTConfig, build_model
from clincot.models.checkpoint_io import load_checkpoint
from clincot.models.language_backbone import LanguageConfig
from clincot.models.multimodal_projector import ProjectorConfig
from clincot.models.vision_backbone import VisionConfig
from clincot.data.preprocess import maybe_load_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
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
    load_checkpoint(model, Path(args.checkpoint), map_location="cpu", strict=False)

    image = maybe_load_image(args.image)
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    prompt = f"Question: {args.question}\nAnswer:"
    toks = tokenizer(prompt, return_tensors="pt").to(device)

    out = model.generate(
        input_ids=toks["input_ids"],
        attention_mask=toks["attention_mask"],
        pixel_values=pixel_values,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
