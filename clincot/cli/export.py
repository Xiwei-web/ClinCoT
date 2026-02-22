from __future__ import annotations

import argparse
from pathlib import Path
import torch
import yaml

from clincot.models.builder import ClinCoTConfig, build_model
from clincot.models.checkpoint_io import load_checkpoint
from clincot.models.language_backbone import LanguageConfig
from clincot.models.multimodal_projector import ProjectorConfig
from clincot.models.vision_backbone import VisionConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
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

    model = build_model(model_cfg)
    load_checkpoint(model, Path(args.checkpoint), map_location="cpu", strict=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"exported: {out}")


if __name__ == "__main__":
    main()
