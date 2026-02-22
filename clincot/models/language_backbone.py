from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
from torch import nn


@dataclass
class LanguageConfig:
    model_name_or_path: str
    freeze: bool = False
    trust_remote_code: bool = False


class LanguageBackbone(nn.Module):
    def __init__(self, cfg: LanguageConfig) -> None:
        super().__init__()
        self.cfg = cfg

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("transformers is required for LanguageBackbone") from exc

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
        )
        if cfg.freeze:
            self.freeze_parameters()

    @property
    def hidden_size(self) -> int:
        hs = getattr(self.model.config, "hidden_size", None)
        if hs is None:
            hs = getattr(self.model.config, "n_embd", None)
        if hs is None:
            raise ValueError("Cannot infer language hidden size from config")
        return int(hs)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_parameters(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_parameters(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, **kwargs: Any) -> Any:
        return self.model(**kwargs)

    @torch.inference_mode()
    def generate(self, **kwargs: Any) -> Any:
        return self.model.generate(**kwargs)
