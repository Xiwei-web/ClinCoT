from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from clincot.data.datamodule import build_preference_dataloader, build_sft_dataloader
from clincot.data.datasets import DatasetConfig
from clincot.models.builder import ClinCoTConfig, build_model
from clincot.models.checkpoint_io import latest_checkpoint, load_checkpoint, save_checkpoint
from clincot.models.language_backbone import LanguageConfig
from clincot.models.multimodal_projector import ProjectorConfig
from clincot.models.vision_backbone import VisionConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main() -> bool:
    return rank() == 0


def init_distributed() -> None:
    if not is_distributed():
        return
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def cleanup_distributed() -> None:
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def dpo_loss(
    pi_chosen_logp: torch.Tensor,
    pi_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
    score_chosen: torch.Tensor | None = None,
    score_rejected: torch.Tensor | None = None,
    use_sdpo_margin: bool = False,
) -> torch.Tensor:
    logits = (pi_chosen_logp - pi_rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
    margin = 0.0
    if use_sdpo_margin and score_chosen is not None and score_rejected is not None:
        margin = score_chosen - score_rejected
    return -torch.nn.functional.logsigmoid(beta * logits - margin)


def _token_logp_from_outputs(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    labels = labels[:, 1:].contiguous()
    logits = logits[:, :-1, :].contiguous()
    mask = labels.ne(ignore_index)

    labels = labels.masked_fill(~mask, 0)
    token_logp = torch.gather(logits.log_softmax(dim=-1), dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["sft", "pref"], default="sft")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_distributed()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ClinCoTConfig(
        vision=VisionConfig(**cfg["model"]["vision"]),
        language=LanguageConfig(**cfg["model"]["language"]),
        projector=ProjectorConfig(**cfg["model"]["projector"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.language.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_processor = AutoImageProcessor.from_pretrained(model_cfg.vision.model_name_or_path)

    model = build_model(model_cfg).to(device)
    ref_model = None

    if args.mode == "pref":
        ref_model = copy.deepcopy(model).eval().to(device)
        for p in ref_model.parameters():
            p.requires_grad = False

    if is_distributed():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    data_cfg = DatasetConfig(**cfg["data"])
    train_cfg = cfg["train"]

    if args.mode == "sft":
        dataloader = build_sft_dataloader(
            data_cfg,
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=int(train_cfg.get("batch_size", 1)),
            num_workers=int(train_cfg.get("num_workers", 4)),
            distributed=is_distributed(),
            shuffle=True,
        )
    else:
        dataloader = build_preference_dataloader(
            data_cfg,
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=int(train_cfg.get("batch_size", 1)),
            num_workers=int(train_cfg.get("num_workers", 4)),
            distributed=is_distributed(),
            shuffle=True,
        )

    model_to_opt = model.module if isinstance(model, DDP) else model
    optimizer = AdamW(model_to_opt.parameters(), lr=float(train_cfg.get("lr", 1e-5)), weight_decay=float(train_cfg.get("weight_decay", 0.0)))

    start_step = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt = latest_checkpoint(output_dir)
        if ckpt is not None:
            state = load_checkpoint(model_to_opt, ckpt, optimizer=optimizer, map_location="cpu", strict=False)
            start_step = int(state.get("step", 0))
            if is_main():
                print(f"Resumed from {ckpt} at step {start_step}")

    epochs = int(train_cfg.get("epochs", 1))
    log_every = int(train_cfg.get("log_every", 20))
    save_every = int(train_cfg.get("save_every", 200))
    beta = float(train_cfg.get("beta", 0.1))
    use_sdpo_margin = bool(train_cfg.get("use_sdpo_margin", True))

    global_step = start_step
    model.train()

    for epoch in range(epochs):
        if is_distributed() and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        iterator = tqdm(dataloader, disable=not is_main(), desc=f"epoch {epoch}")
        for batch in iterator:
            batch = move_batch_to_device(batch, device)

            if args.mode == "sft":
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    pixel_values=batch["pixel_values"],
                )
                loss = outputs.loss
            else:
                chosen_outputs = model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"],
                    labels=batch["chosen_labels"],
                    pixel_values=batch["pixel_values"],
                )
                rejected_pixel = batch.get("rejected_pixel_values", batch["pixel_values"])
                rejected_outputs = model(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"],
                    labels=batch["rejected_labels"],
                    pixel_values=rejected_pixel,
                )

                with torch.no_grad():
                    ref_chosen = ref_model(
                        input_ids=batch["chosen_input_ids"],
                        attention_mask=batch["chosen_attention_mask"],
                        labels=batch["chosen_labels"],
                        pixel_values=batch["pixel_values"],
                    )
                    ref_rejected = ref_model(
                        input_ids=batch["rejected_input_ids"],
                        attention_mask=batch["rejected_attention_mask"],
                        labels=batch["rejected_labels"],
                        pixel_values=rejected_pixel,
                    )

                pi_c = _token_logp_from_outputs(chosen_outputs.logits, batch["chosen_labels"])
                pi_r = _token_logp_from_outputs(rejected_outputs.logits, batch["rejected_labels"])
                ref_c = _token_logp_from_outputs(ref_chosen.logits, batch["chosen_labels"])
                ref_r = _token_logp_from_outputs(ref_rejected.logits, batch["rejected_labels"])

                losses = dpo_loss(
                    pi_c,
                    pi_r,
                    ref_c,
                    ref_r,
                    beta=beta,
                    score_chosen=batch.get("score_chosen"),
                    score_rejected=batch.get("score_rejected"),
                    use_sdpo_margin=use_sdpo_margin,
                )
                loss = losses.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1

            if is_main() and global_step % log_every == 0:
                iterator.set_postfix(loss=float(loss.item()), step=global_step)

            if is_main() and global_step % save_every == 0:
                save_checkpoint(model_to_opt, output_dir=output_dir, step=global_step, optimizer=optimizer)

    if is_main():
        save_checkpoint(model_to_opt, output_dir=output_dir, step=global_step, optimizer=optimizer)

    cleanup_distributed()


if __name__ == "__main__":
    main()
