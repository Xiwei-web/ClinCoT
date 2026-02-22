from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class SDPOHyperParams:
    beta: float = 0.1
    label_smoothing: float = 0.0
    reference_free: bool = False


def _build_logits(
    pi_chosen_logp: torch.Tensor,
    pi_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    reference_free: bool = False,
) -> torch.Tensor:
    pi_logratios = pi_chosen_logp - pi_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    if reference_free:
        ref_logratios = torch.zeros_like(pi_logratios)
    return pi_logratios - ref_logratios


def compute_dpo_loss(
    pi_chosen_logp: torch.Tensor,
    pi_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = _build_logits(
        pi_chosen_logp,
        pi_rejected_logp,
        ref_chosen_logp,
        ref_rejected_logp,
        reference_free=reference_free,
    )

    pos = -F.logsigmoid(beta * logits)
    if label_smoothing > 0:
        neg = -F.logsigmoid(-beta * logits)
        losses = (1 - label_smoothing) * pos + label_smoothing * neg
    else:
        losses = pos

    chosen_rewards = beta * (pi_chosen_logp - ref_chosen_logp).detach()
    rejected_rewards = beta * (pi_rejected_logp - ref_rejected_logp).detach()
    return losses, chosen_rewards, rejected_rewards


def compute_sdpo_loss(
    pi_chosen_logp: torch.Tensor,
    pi_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    score_chosen: torch.Tensor,
    score_rejected: torch.Tensor,
    beta: float = 0.1,
    score_scale: float = 1.0,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = _build_logits(
        pi_chosen_logp,
        pi_rejected_logp,
        ref_chosen_logp,
        ref_rejected_logp,
        reference_free=reference_free,
    )

    margin = (score_chosen - score_rejected) * score_scale
    pos = -F.logsigmoid(beta * logits - margin)

    if label_smoothing > 0:
        neg = -F.logsigmoid(-(beta * logits - margin))
        losses = (1 - label_smoothing) * pos + label_smoothing * neg
    else:
        losses = pos

    chosen_rewards = beta * (pi_chosen_logp - ref_chosen_logp).detach()
    rejected_rewards = beta * (pi_rejected_logp - ref_rejected_logp).detach()
    return losses, chosen_rewards, rejected_rewards
