from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn


@dataclass
class VisionConfig:
    model_name_or_path: str
    feature_pool: Literal["patch", "cls", "mean"] = "patch"
    freeze: bool = False


class VisionBackbone(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("transformers is required for VisionBackbone") from exc

        self.model = AutoModel.from_pretrained(cfg.model_name_or_path)
        if cfg.freeze:
            self.freeze_parameters()

    @property
    def hidden_size(self) -> int:
        hs = getattr(self.model.config, "hidden_size", None)
        if hs is None:
            hs = getattr(self.model.config, "vision_width", None)
        if hs is None:
            raise ValueError("Cannot infer vision hidden size from config")
        return int(hs)

    def freeze_parameters(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_parameters(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
        x = outputs.last_hidden_state  # [B, N, C]

        if self.cfg.feature_pool == "patch":
            return x
        if self.cfg.feature_pool == "cls":
            return x[:, :1, :]
        if self.cfg.feature_pool == "mean":
            return x.mean(dim=1, keepdim=True)
        raise ValueError(f"Unknown feature_pool={self.cfg.feature_pool}")
