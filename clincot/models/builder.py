from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn

from .language_backbone import LanguageBackbone, LanguageConfig
from .multimodal_projector import ProjectorConfig, build_projector
from .vision_backbone import VisionBackbone, VisionConfig


@dataclass
class ClinCoTConfig:
    vision: VisionConfig
    language: LanguageConfig
    projector: ProjectorConfig
    ignore_index: int = -100
    image_token_strategy: str = "prepend"


class ClinCoTModel(nn.Module):
    """Minimal multimodal model wrapper for ClinCoT.

    Current strategy:
    - Encode image into visual tokens.
    - Project visual tokens to language hidden size.
    - Prepend projected tokens before text embeddings.
    """

    def __init__(self, cfg: ClinCoTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.vision = VisionBackbone(cfg.vision)

        projector_cfg = cfg.projector
        projector_cfg.in_dim = self.vision.hidden_size

        self.language = LanguageBackbone(cfg.language)
        projector_cfg.out_dim = self.language.hidden_size

        self.projector = build_projector(projector_cfg)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.vision(pixel_values)
        return self.projector(visual_tokens)

    def _fuse_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        image_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        text_embeds = self.language.get_input_embeddings()(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.long)

        if self.cfg.image_token_strategy != "prepend":
            raise NotImplementedError("Only prepend strategy is implemented")

        fused_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        image_mask = torch.ones(
            (attention_mask.size(0), image_embeds.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        fused_attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        fused_labels = None
        if labels is not None:
            image_labels = torch.full(
                (labels.size(0), image_embeds.size(1)),
                self.cfg.ignore_index,
                dtype=labels.dtype,
                device=labels.device,
            )
            fused_labels = torch.cat([image_labels, labels], dim=1)

        return fused_embeds, fused_attention_mask, fused_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        image_embeds = self.encode_image(pixel_values)
        fused_embeds, fused_attention_mask, fused_labels = self._fuse_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_embeds=image_embeds,
        )
        return self.language(
            inputs_embeds=fused_embeds,
            attention_mask=fused_attention_mask,
            labels=fused_labels,
            **kwargs,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        image_embeds = self.encode_image(pixel_values)
        fused_embeds, fused_attention_mask, _ = self._fuse_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            image_embeds=image_embeds,
        )
        return self.language.generate(
            inputs_embeds=fused_embeds,
            attention_mask=fused_attention_mask,
            **kwargs,
        )


def build_model(cfg: ClinCoTConfig) -> ClinCoTModel:
    return ClinCoTModel(cfg)
