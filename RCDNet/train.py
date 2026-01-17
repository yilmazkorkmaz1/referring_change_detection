import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

import argparse
import importlib
import logging
import os
import time

# Suppress PyTorch DDP temporary file messages
logging.getLogger("torch.distributed.nn.jit.instantiator").setLevel(logging.ERROR)
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import logging as hf_logging

from dataloader.dataloader import get_train_loader, get_val_loader
from models.builder import EncoderDecoder as SegModel
from rcd_eval import evaluate_referring_cd

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


@dataclass
class TrainState:
    epoch: int
    step: int
    best_score: float


def load_config(config_path: str):
    """
    config_path example: 'configs.config_second'
    """
    mod = importlib.import_module(config_path)
    if not hasattr(mod, "config"):
        raise ValueError(f"{config_path} must expose `config`")
    return mod.config


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, state: TrainState):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": state.epoch,
            "step": state.step,
            "best_score": state.best_score,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Referring Change Detection (RCDNet)")
    parser.add_argument("--config", type=str, default="configs.config_second", help="Config file path")
    parser.add_argument("--data_root", type=str, default=None, help="Override config.root_folder")
    parser.add_argument("--output_dir", type=str, default="runs", help="Where to write logs/checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.nepochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override config.lr")
    parser.add_argument("--num_workers", type=int, default=None, help="Override config.num_workers")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--amp", action="store_true", help="Use torch autocast + GradScaler")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume (loads model + optimizer + epoch)")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to checkpoint .pt/.safetensors to init model weights (fresh optimizer, starts epoch 1)")
    parser.add_argument("--train_text_encoder", action="store_true", help="Fine-tune CLIP text encoder")
    parser.add_argument("--eval_threshold", type=float, default=0.5, help="Sigmoid threshold used for eval metrics")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="RCDNet", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team (optional)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.root_folder = args.data_root
    if args.epochs is not None:
        cfg.nepochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    # Use normalization values from config (not cached JSON files)
    cfg.use_cached_norm = getattr(cfg, "use_cached_norm", False)
    cfg.use_single_normalization = getattr(cfg, "use_single_normalization", False)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision=("fp16" if args.amp else "no"), kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    if accelerator.is_main_process:
        hf_logging.set_verbosity_warning()
    else:
        hf_logging.set_verbosity_error()
        logging.getLogger("torch.distributed").setLevel(logging.ERROR)
        logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\\..*")

    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("torch.distributed.nn.jit.templates.remote_module_template").setLevel(logging.ERROR)

    torch.manual_seed(getattr(cfg, "seed", 0))
    np.random.seed(getattr(cfg, "seed", 0))

    train_loader = get_train_loader(None, None, cfg) 
    val_dataset = get_val_loader(cfg)

    criterion = nn.BCEWithLogitsLoss()
    model = SegModel(cfg=cfg, criterion=criterion, criterion2=None, norm_layer=nn.BatchNorm2d).to(device)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    llm = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    llm.requires_grad_(args.train_text_encoder)

    params = list(model.parameters()) + (list(llm.parameters()) if args.train_text_encoder else [])
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=getattr(cfg, "weight_decay", 0.01))

    run_name = args.wandb_run_name or getattr(cfg, "trial_name", f"{cfg.dataset_name}_{cfg.backbone}_{cfg.decoder}")
    run_dir = os.path.join(args.output_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    state = TrainState(epoch=1, step=0, best_score=0.0)

    wandb_run = None
    if args.wandb and accelerator.is_main_process:
        import wandb 

        cfg_dict = dict(cfg)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,  
            config={
                **cfg_dict,
                "args.config": args.config,
                "args.data_root": args.data_root,
                "args.output_dir": args.output_dir,
                "args.amp": args.amp,
                "args.eval_every": args.eval_every,
                "args.train_text_encoder": args.train_text_encoder,
                "device": str(device),
                "accelerate.num_processes": accelerator.num_processes,
            },
        )

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        state.epoch = int(ckpt.get("epoch", 0)) + 1
        state.step = int(ckpt.get("step", 0))
        state.best_score = float(ckpt.get("best_score", 0.0))
        accelerator.print(f"Resumed from {args.resume} at epoch {state.epoch}")

    pretrained_path = args.pretrained or getattr(cfg, "pretrained_checkpoint", None)
    if pretrained_path:
        if pretrained_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_path)
        else:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)  
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            accelerator.print(f"[pretrained] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            accelerator.print(f"[pretrained] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        accelerator.print(f"Loaded pretrained weights from {pretrained_path}")

    model, llm, optimizer, train_loader = accelerator.prepare(model, llm, optimizer, train_loader)

    for epoch in range(state.epoch, cfg.nepochs + 1):
        model.train()
        llm.train(args.train_text_encoder)

        pbar = tqdm(
            train_loader,
            desc=f"train epoch {epoch}/{cfg.nepochs}",
            disable=not accelerator.is_local_main_process,
        )
        running_loss = 0.0
        running_n = 0
        for batch in pbar:
            A = batch["A"].to(device, non_blocking=True)
            B = batch["B"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            captions = [str(c).lower() for c in batch["text_captions"]]

            inputs = tokenizer(
                list(captions),
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            with accelerator.autocast():
                caption_emb = llm(**inputs).last_hidden_state
                loss = model(A, B, gt, captions=caption_emb)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()

            state.step += 1
            loss_item = float(loss.item())
            running_loss += loss_item
            running_n += 1
            pbar.set_postfix(loss=f"{loss_item:.4f}")

            if wandb_run is not None:
                lr = float(optimizer.param_groups[0]["lr"])
                wandb_run.log(
                    {
                        "train/loss": loss_item,
                        "train/lr": lr,
                        "train/epoch": epoch,
                    },
                    step=state.step,
                )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            save_checkpoint(
                os.path.join(ckpt_dir, "last.pt"),
                unwrapped_model,
                optimizer,
                TrainState(epoch, state.step, state.best_score),
            )

        if args.eval_every > 0 and (epoch % args.eval_every == 0):
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_llm = accelerator.unwrap_model(llm)
                _, score, metrics = evaluate_referring_cd(
                    dataset=val_dataset,
                    network=unwrapped_model,
                    llm_model=unwrapped_llm,
                    tokenizer=tokenizer,
                    num_classes=cfg.num_classes,
                    device=device,
                    threshold=args.eval_threshold,
                    lowercase_captions=True,
                    return_metrics=True,
                )
                accelerator.print(f"[epoch {epoch}] val Score: {score:.6f}")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "val/epoch": epoch,
                            "val/Score": float(score),
                            "val/best_Score": float(state.best_score),
                            "train/epoch_loss": float(running_loss / max(running_n, 1)),
                            **{f"val/{k}": v for k, v in metrics.items() if k != "Score"},
                        },
                        step=state.step,
                    )
                if score > state.best_score:
                    state.best_score = score
                    save_checkpoint(
                        os.path.join(ckpt_dir, "best.pt"),
                        unwrapped_model,
                        optimizer,
                        TrainState(epoch, state.step, state.best_score),
                    )

        if accelerator.is_main_process:
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "run.txt"), "a") as f:
                f.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} epoch={epoch} step={state.step} best_score={state.best_score:.6f}\n"
                )

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()

