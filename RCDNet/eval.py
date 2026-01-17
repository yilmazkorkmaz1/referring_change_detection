"""
Evaluation script for Referring Change Detection (RCDNet).
"""
import warnings
# Suppress FutureWarnings from timm and huggingface_hub before they're imported
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Compatibility shim for older transformers + torch 2.0
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_pytree_node"):
    torch.utils._pytree.register_pytree_node = torch.utils._pytree._register_pytree_node

import argparse
import importlib
import logging
import os

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

from dataloader.dataloader import get_val_loader
from models.builder import EncoderDecoder as SegModel
from rcd_eval import evaluate_referring_cd

# Silence expected HF warning about unused vision weights when loading CLIPTextModel
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def load_model_state_dict(path: str):
    """Load state dict from .pt/.pth or .safetensors file."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")


def load_config(config_path: str):
    """Load config module, e.g. 'configs.config_second'."""
    mod = importlib.import_module(config_path)
    if not hasattr(mod, "config"):
        raise ValueError(f"{config_path} must expose `config`")
    return mod.config


def main():
    parser = argparse.ArgumentParser(description="Evaluate RCDNet")
    parser.add_argument("--config", type=str, default="configs.config_second",
                        help="Config module path, e.g. configs.config_second")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt, .pth, or .safetensors)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override config.root_folder")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split: train/val/test/evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for binary prediction")
    parser.add_argument("--aggregation", type=str, default="overwrite",
                        choices=["overwrite", "max_prob"],
                        help="Prediction aggregation strategy")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision (not used in eval, placeholder)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.root_folder = args.data_root
    # Set val_split to the requested split (get_val_loader uses val_split, not split)
    cfg.val_split = args.split

    cfg.use_cached_norm = getattr(cfg, "use_cached_norm", False)
    cfg.use_single_normalization = getattr(cfg, "use_single_normalization", True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = SegModel(cfg=cfg, criterion=None, norm_layer=nn.BatchNorm2d).to(device)
    state_dict = load_model_state_dict(args.checkpoint)

    # Handle 'model' key if checkpoint was saved with optimizer etc.
    if "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load CLIP text encoder
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    llm = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    llm.eval()

    # Load dataset
    dataset = get_val_loader(cfg)
    print(f"Loaded {len(dataset)} samples from {cfg.root_folder} (split: {cfg.val_split})")

    # Run evaluation
    with torch.no_grad():
        _, score, metrics = evaluate_referring_cd(
            dataset=dataset,
            network=model,
            llm_model=llm,
            tokenizer=tokenizer,
            num_classes=cfg.num_classes,
            device=device,
            threshold=args.threshold,
            lowercase_captions=True,
            aggregation=args.aggregation,
            return_metrics=True,
            show_progress=True,
        )

    print(f"\n{'='*50}")
    print(f"Final Score: {score:.6f}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if not k.startswith("IoU_"):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
