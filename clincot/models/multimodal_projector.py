from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class ProjectorConfig:
    projector_type: str = "mlp2x_gelu"
    in_dim: int = 1024
    out_dim: int = 4096
    hidden_dim: int = 4096
    dropout: float = 0.0


class IdentityProjector(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def build_projector(cfg: ProjectorConfig) -> nn.Module:
    ptype = cfg.projector_type.lower()
    if ptype in {"identity", "none"}:
        if cfg.in_dim != cfg.out_dim:
            return LinearProjector(cfg.in_dim, cfg.out_dim, cfg.dropout)
        return IdentityProjector()
    if ptype in {"linear"}:
        return LinearProjector(cfg.in_dim, cfg.out_dim, cfg.dropout)
    if ptype in {"mlp", "mlp2x_gelu", "mlp2x"}:
        return MLPProjector(cfg.in_dim, cfg.hidden_dim, cfg.out_dim, cfg.dropout)
    raise ValueError(f"Unsupported projector_type={cfg.projector_type}")
