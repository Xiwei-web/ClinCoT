from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json
import torch


def save_checkpoint(
    model: torch.nn.Module,
    output_dir: str | Path,
    step: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    extra_state: Optional[dict[str, Any]] = None,
) -> Path:
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / f"checkpoint-{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra_state is not None:
        payload["extra_state"] = extra_state

    torch.save(payload, ckpt_dir / "training_state.pt")

    meta = {"step": step, "path": str(ckpt_dir)}
    with open(output_dir / "latest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return ckpt_dir


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path / "training_state.pt", map_location=map_location)

    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    return state


def latest_checkpoint(output_dir: str | Path) -> Optional[Path]:
    output_dir = Path(output_dir)
    latest_file = output_dir / "latest.json"
    if not latest_file.exists():
        return None
    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Path(data["path"])
