from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from .clincot_pipeline import ClinCoTBatch, ClinCoTPipeline


@dataclass
class TrainerConfig:
    epochs: int = 1
    log_every: int = 20
    save_every: int = 200
    grad_accum_steps: int = 1


class ClinCoTTrainer:
    """Thin trainer abstraction for method-level reuse."""

    def __init__(
        self,
        pipeline: ClinCoTPipeline,
        optimizer: torch.optim.Optimizer,
        cfg: TrainerConfig,
        checkpoint_callback=None,
        is_main_process: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.optimizer = optimizer
        self.cfg = cfg
        self.checkpoint_callback = checkpoint_callback
        self.is_main_process = is_main_process
        self.global_step = 0

    def _to_device(self, batch: dict, device: torch.device) -> dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(device) if torch.is_tensor(v) else v
        return out

    def train(self, dataloader, mode: str, device: torch.device) -> None:
        self.pipeline.model.train()
        iterator = range(self.cfg.epochs)

        for epoch in iterator:
            bar = tqdm(dataloader, disable=not self.is_main_process, desc=f"epoch {epoch}")
            for step_idx, batch in enumerate(bar):
                batch = self._to_device(batch, device)
                loss = self.pipeline.forward(ClinCoTBatch(mode=mode, tensors=batch))
                loss = loss / max(self.cfg.grad_accum_steps, 1)
                loss.backward()

                if (step_idx + 1) % self.cfg.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if self.is_main_process and self.global_step % self.cfg.log_every == 0:
                        bar.set_postfix(loss=float(loss.item() * self.cfg.grad_accum_steps), step=self.global_step)

                    if (
                        self.is_main_process
                        and self.checkpoint_callback is not None
                        and self.global_step % self.cfg.save_every == 0
                    ):
                        self.checkpoint_callback(self.global_step)
