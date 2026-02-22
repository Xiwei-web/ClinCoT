from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .sdpo_loss import compute_dpo_loss, compute_sdpo_loss


@dataclass
class ClinCoTBatch:
    mode: str  # "sft" or "pref"
    tensors: dict[str, Any]


class ClinCoTPipeline:
    """Method-level pipeline wrapper.

    It centralizes forward/loss logic and keeps CLI thin.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: Optional[torch.nn.Module] = None,
        beta: float = 0.1,
        use_sdpo_margin: bool = True,
        score_scale: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        self.use_sdpo_margin = use_sdpo_margin
        self.score_scale = score_scale
        self.ignore_index = ignore_index

    def _token_logp(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        mask = labels.ne(self.ignore_index)
        labels = labels.masked_fill(~mask, 0)
        token_logp = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
        return (token_logp * mask).sum(dim=-1)

    def forward(self, batch: ClinCoTBatch) -> torch.Tensor:
        t = batch.tensors
        if batch.mode == "sft":
            outputs = self.model(
                input_ids=t["input_ids"],
                attention_mask=t["attention_mask"],
                labels=t["labels"],
                pixel_values=t["pixel_values"],
            )
            return outputs.loss

        chosen_outputs = self.model(
            input_ids=t["chosen_input_ids"],
            attention_mask=t["chosen_attention_mask"],
            labels=t["chosen_labels"],
            pixel_values=t["pixel_values"],
        )

        rejected_pixel = t.get("rejected_pixel_values", t["pixel_values"])
        rejected_outputs = self.model(
            input_ids=t["rejected_input_ids"],
            attention_mask=t["rejected_attention_mask"],
            labels=t["rejected_labels"],
            pixel_values=rejected_pixel,
        )

        pi_c = self._token_logp(chosen_outputs.logits, t["chosen_labels"])
        pi_r = self._token_logp(rejected_outputs.logits, t["rejected_labels"])

        if self.ref_model is None:
            raise ValueError("ref_model is required for preference optimization")

        with torch.no_grad():
            ref_c_outputs = self.ref_model(
                input_ids=t["chosen_input_ids"],
                attention_mask=t["chosen_attention_mask"],
                labels=t["chosen_labels"],
                pixel_values=t["pixel_values"],
            )
            ref_r_outputs = self.ref_model(
                input_ids=t["rejected_input_ids"],
                attention_mask=t["rejected_attention_mask"],
                labels=t["rejected_labels"],
                pixel_values=rejected_pixel,
            )

        ref_c = self._token_logp(ref_c_outputs.logits, t["chosen_labels"])
        ref_r = self._token_logp(ref_r_outputs.logits, t["rejected_labels"])

        if self.use_sdpo_margin and "score_chosen" in t and "score_rejected" in t:
            losses, _, _ = compute_sdpo_loss(
                pi_c,
                pi_r,
                ref_c,
                ref_r,
                score_chosen=t["score_chosen"],
                score_rejected=t["score_rejected"],
                beta=self.beta,
                score_scale=self.score_scale,
            )
        else:
            losses, _, _ = compute_dpo_loss(
                pi_c,
                pi_r,
                ref_c,
                ref_r,
                beta=self.beta,
            )

        return losses.mean()
