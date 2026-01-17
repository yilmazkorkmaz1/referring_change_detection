#!/usr/bin/env python
"""
Script to download RCDNet synthetic datasets from Hugging Face Hub.

IMPORTANT: The synthetic datasets do NOT include the A/ (pre-change) images.
You have two options:
1. Download the original dataset and symlink/copy the A/ folder (recommended)
2. Use synthetic_A from B/ (every 10th image generated with no-change prompts)

Usage:
    # Option 1: Download SECOND synthetic with original A (recommended)
    python scripts/download_from_hf.py \
        --repo_name yilmazkorkmaz/Synthetic_RCD_1 \
        --local_dir data/second_synthetic \
        --original_a_path /path/to/second_dataset/train/A

    # Option 2: Download CNAM-CD synthetic with synthetic_A from B/
    python scripts/download_from_hf.py \
        --repo_name yilmazkorkmaz/Synthetic_RCD_2 \
        --local_dir data/cnamcd_synthetic \
        --use_synthetic_a
"""

import argparse
import os
import shutil
from pathlib import Path


def download_dataset(repo_name: str, local_dir: str) -> Path:
    """Download a dataset from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return None

    print(f"Downloading dataset from: {repo_name}")
    print(f"Saving to: {local_dir}")
    
    local_path = snapshot_download(
        repo_id=repo_name,
        repo_type="dataset",
        local_dir=local_dir,
    )
    
    print(f"\n✅ Dataset downloaded to: {local_path}")
    return Path(local_path)


def setup_a_folder(local_path: Path, original_a_path: str = None, use_synthetic_a: bool = False):
    """Set up the A/ folder from original dataset or synthetic_A from B/."""
    train_path = local_path / "train"
    a_path = train_path / "A"
    
    if a_path.exists() or a_path.is_symlink():
        print(f"A/ folder already exists at {a_path}")
        return
    
    if use_synthetic_a:
        # Create A/ folder with symlinks to B/ images referenced in train_synthetic_A.txt
        synthetic_txt = train_path / "train_synthetic_A.txt"
        b_path = train_path / "B"
        
        if synthetic_txt.exists() and b_path.exists():
            print("Setting up A/ from synthetic_A images in B/...")
            a_path.mkdir(parents=True, exist_ok=True)
            
            # Get unique synthetic_A filenames
            with open(synthetic_txt) as f:
                unique_files = set(line.strip() for line in f if line.strip())
            
            # Create symlinks from A/ to B/
            for filename in unique_files:
                src = b_path / filename
                dst = a_path / filename
                if src.exists() and not dst.exists():
                    os.symlink(src.resolve(), dst)
            
            # Copy train_synthetic_A.txt to train_A.txt
            a_txt = train_path / "train_A.txt"
            shutil.copy2(synthetic_txt, a_txt)
            
            print(f"✅ Created A/ with {len(unique_files)} symlinks to B/")
            print(f"✅ Copied train_synthetic_A.txt to train_A.txt")
        else:
            print("⚠️  train_synthetic_A.txt or B/ folder not found.")
    
    elif original_a_path:
        original_a = Path(original_a_path)
        if original_a.exists():
            print(f"Creating symlink A/ -> {original_a}")
            os.symlink(original_a.resolve(), a_path)
            print(f"✅ A/ linked to original dataset")
        else:
            print(f"⚠️  Original A/ path does not exist: {original_a}")
            print("   Please download the original dataset first.")
    
    else:
        print("\n" + "="*70)
        print("⚠️  IMPORTANT: A/ folder not set up!")
        print("="*70)
        print("\nThe synthetic dataset does not include pre-change images (A/).")
        print("You need to set this up. Re-run with one of these options:\n")
        print("Option 1: Link to original dataset (RECOMMENDED)")
        print("  --original_a_path /path/to/original_dataset/train/A")
        print("\nOption 2: Use synthetic_A from B/ (no-change images)")
        print("  --use_synthetic_a")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download synthetic dataset from Hugging Face Hub")
    parser.add_argument("--repo_name", type=str, required=True,
                       help="Hugging Face repo name (e.g., 'rcdnet/second-synthetic-cd')")
    parser.add_argument("--local_dir", type=str, required=True,
                       help="Local directory to save the dataset")
    parser.add_argument("--original_a_path", type=str, default=None,
                       help="Path to original dataset's A/ folder (to create symlink)")
    parser.add_argument("--use_synthetic_a", action="store_true",
                       help="Use synthetic_A from B/ (no-change images, every 10th)")
    args = parser.parse_args()

    # Download the dataset
    local_path = download_dataset(args.repo_name, args.local_dir)
    
    if local_path is None:
        return
    
    # Verify structure
    train_path = local_path / "train"
    if (train_path / "B").exists():
        num_b = len(list((train_path / "B").glob("*.png")))
        print(f"Found {num_b} B (post-change) images")
    if (train_path / "gt").exists():
        num_gt = len(list((train_path / "gt").glob("*.png")))
        print(f"Found {num_gt} ground truth masks")
    if (train_path / "train_synthetic_A.txt").exists():
        with open(train_path / "train_synthetic_A.txt") as f:
            num_synthetic_a = len(set(line.strip() for line in f if line.strip()))
        print(f"Found {num_synthetic_a} unique synthetic_A images (in B/)")
    
    # Set up A/ folder
    setup_a_folder(local_path, args.original_a_path, args.use_synthetic_a)


if __name__ == "__main__":
    main()
