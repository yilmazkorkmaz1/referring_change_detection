#!/usr/bin/env python
"""
Script to upload RCDNet synthetic datasets to Hugging Face Hub.

IMPORTANT: This script uploads ONLY the synthetic data (B/, gt/, txt files).
The A/ folder images come from the original datasets and should NOT be re-uploaded.

Usage:
    # First, login to Hugging Face:
    huggingface-cli login

    # Upload SECOND synthetic dataset:
    python scripts/upload_to_hf.py \
        --dataset_path /mnt/store/ykorkma1/datasets/second_dataset_with_1ch \
        --repo_name yilmazkorkmaz/Synthetic_RCD_1 \
        --dataset_name "Synthetic RCD Dataset 1 (SECOND)" \
        --original_dataset_url "https://captain-whu.github.io/SCD/"

    # Upload CNAM-CD synthetic dataset:
    python scripts/upload_to_hf.py \
        --dataset_path /mnt/store/ykorkma1/datasets/cnamcd_dataset_with_1ch \
        --repo_name yilmazkorkmaz/Synthetic_RCD_2 \
        --dataset_name "Synthetic RCD Dataset 2 (CNAM-CD)" \
        --original_dataset_url "https://github.com/Chen-Yang-Liu/CNAM-CD"
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def create_dataset_card(
    dataset_name: str, 
    num_samples: int, 
    num_synthetic_a: int,
    class_names: list,
    original_dataset_url: str,
    original_dataset_name: str,
) -> str:
    """Create a README.md for the Hugging Face dataset."""
    return f'''---
license: mit
task_categories:
  - image-segmentation
tags:
  - change-detection
  - remote-sensing
  - semantic-segmentation
  - synthetic-data
size_categories:
  - 10K<n<100K
---

# {dataset_name}

This is a **synthetic** change detection dataset for training [RCDNet](https://github.com/your-username/RCDNet).

## Important: Pre-change Images (A/)

**This dataset does NOT include the pre-change images (A/)** because they come from the original {original_dataset_name} dataset. You have two options:

### Option 1: Use Original Dataset (Recommended)

1. Download the original {original_dataset_name} dataset from: {original_dataset_url}
2. Create a symlink to the A/ folder:

```bash
# After downloading this dataset and the original
ln -s /path/to/original/{original_dataset_name.lower().replace("-", "")}/train/A train/A
```

### Option 2: Use Synthetic A from B/ (No Original Dataset Needed)

The `train_synthetic_A.txt` file maps each sample to a "no-change" image already in B/ (every 10th: 0.png, 10.png, 20.png...). These {num_synthetic_a} unique images were generated with "no-change" prompts and can serve as A images:

```bash
# Using RCDNet helper script
python scripts/download_from_hf.py \\
  --repo_name your-username/dataset-name \\
  --local_dir data/dataset_synthetic \\
  --use_synthetic_a
```

## Dataset Description

- **Synthetic B samples**: {num_samples}
- **Synthetic A samples**: {num_synthetic_a} (no-change images in B/, every 10th)
- **Image Size**: 512x512
- **Format**: PNG images
- **Classes**: {len(class_names)}

### What's Included

| Folder/File | Description |
|-------------|-------------|
| `train/B/` | Synthetic post-change images ({num_samples} samples) |
| `train/gt/` | Ground truth semantic change masks |
| `train_A.txt` | File list mapping to original dataset A/ images |
| `train_B.txt` | File list for synthetic B/ images |
| `train_gt.txt` | File list for ground truth masks |
| `train_synthetic_A.txt` | File list for no-change samples (every 10th in B/) |

### What's NOT Included

| Folder | Reason |
|--------|--------|
| `train/A/` | Pre-change images from original {original_dataset_name} dataset |

### Class Names

```
{chr(10).join(f"{i}: {name}" for i, name in enumerate(class_names))}
```

## Image Enumeration Scheme

The synthetic images follow a specific naming convention that encodes both the pre-change image (A) and the change class:

```
image_index = (A_index * 10) + class_index
```

| Image | A Index | Class Index | Change Class |
|-------|---------|-------------|--------------|
| 0.png | 0 | 0 | non-change |
| 1.png | 0 | 1 | impervious surface |
| 2.png | 0 | 2 | bare ground |
| ... | ... | ... | ... |
| 9.png | 0 | 9 | playground |
| 10.png | 1 | 0 | non-change |
| 11.png | 1 | 1 | impervious surface |
| ... | ... | ... | ... |

**Key formulas:**
- `class_index = image_index % 10` → determines which class changed
- `A_index = image_index // 10` → determines which pre-change image was used
- Images where `image_index % 10 == 0` are "no-change" images (synthetic_A)

## Dataset Structure After Setup

```
dataset/
├── train/
│   ├── A/             # From original dataset OR created from synthetic_A
│   ├── B/             # Synthetic post-change images (included)
│   └── gt/            # Ground truth masks (included)
├── train_A.txt
├── train_B.txt
├── train_gt.txt
└── train_synthetic_A.txt
```

## Quick Setup

### Using RCDNet Helper Script

```bash
# Option 1: Download and link to original dataset (recommended)
python scripts/download_from_hf.py \\
  --repo_name your-username/{original_dataset_name.lower().replace("-", "")}-synthetic-cd \\
  --local_dir data/{original_dataset_name.lower().replace("-", "")}_synthetic \\
  --original_a_path /path/to/original_dataset/train/A

# Option 2: Use synthetic A from B/ (no original dataset needed)
python scripts/download_from_hf.py \\
  --repo_name your-username/{original_dataset_name.lower().replace("-", "")}-synthetic-cd \\
  --local_dir data/{original_dataset_name.lower().replace("-", "")}_synthetic \\
  --use_synthetic_a
```

### Manual Setup

```bash
# 1. Download this dataset
huggingface-cli download your-username/dataset-name --repo-type dataset --local-dir data/synthetic_dataset

# 2a. Link to original dataset (recommended)
ln -s /path/to/original/train/A data/synthetic_dataset/train/A

# 2b. OR create A/ from synthetic_A (no-change images in B/)
mkdir -p data/synthetic_dataset/train/A
# Copy unique synthetic_A images from B/ to A/
for f in $(sort -u data/synthetic_dataset/train/train_synthetic_A.txt); do
  ln -s ../B/$f data/synthetic_dataset/train/A/$f
done
cp data/synthetic_dataset/train/train_synthetic_A.txt data/synthetic_dataset/train/train_A.txt

# 3. Train with RCDNet
accelerate launch train.py --config configs.config_second_synthetic --wandb --amp
```

## Citation

If you use this dataset, please cite both RCDNet and the original {original_dataset_name} dataset.

```bibtex
@inproceedings{{korkmaz2026referring,
  title     = {{Referring Change Detection in Remote Sensing Imagery}},
  author    = {{Korkmaz, Yilmaz and Paranjape, Jay N. and de Melo, Celso M. and Patel, Vishal M.}},
  booktitle = {{Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}},
  year      = {{2026}}
}}
```

## License

This synthetic data is released under the MIT License. Please also respect the license of the original {original_dataset_name} dataset.
'''


def count_samples(dataset_path: Path) -> tuple:
    """Count number of samples in B/ and unique synthetic_A."""
    train_path = dataset_path / "train"
    
    num_b = 0
    if (train_path / "B").exists():
        num_b = len(list((train_path / "B").glob("*.png")))
    
    num_synthetic_a = 0
    synthetic_txt = train_path / "train_synthetic_A.txt"
    if synthetic_txt.exists():
        with open(synthetic_txt) as f:
            unique_files = set(line.strip() for line in f if line.strip())
            num_synthetic_a = len(unique_files)
    
    return num_b, num_synthetic_a


def prepare_upload_folder(dataset_path: Path, temp_dir: Path, train_only: bool = True) -> Path:
    """
    Prepare a clean folder for upload, excluding A/ (symlink to original dataset).
    Only copies images that are referenced in the txt files.
    """
    if train_only:
        train_src = dataset_path / "train"
        train_dst = temp_dir / "train"
    else:
        train_src = dataset_path
        train_dst = temp_dir
    
    train_dst.mkdir(parents=True, exist_ok=True)
    
    # Read the txt files to get the list of images to upload
    b_txt = train_src / "train_B.txt"
    gt_txt = train_src / "train_gt.txt"
    
    b_files = set()
    gt_files = set()
    
    if b_txt.exists():
        with open(b_txt) as f:
            b_files = set(line.strip() for line in f if line.strip())
    
    if gt_txt.exists():
        with open(gt_txt) as f:
            gt_files = set(line.strip() for line in f if line.strip())
    
    # Copy only referenced B/ images
    if (train_src / "B").exists() and b_files:
        print(f"  Copying {len(b_files)} images from B/ folder...")
        (train_dst / "B").mkdir(parents=True, exist_ok=True)
        for filename in b_files:
            src = train_src / "B" / filename
            dst = train_dst / "B" / filename
            if src.exists():
                shutil.copy2(src, dst)
    
    # Copy only referenced gt/ images
    if (train_src / "gt").exists() and gt_files:
        print(f"  Copying {len(gt_files)} images from gt/ folder...")
        (train_dst / "gt").mkdir(parents=True, exist_ok=True)
        for filename in gt_files:
            src = train_src / "gt" / filename
            dst = train_dst / "gt" / filename
            if src.exists():
                shutil.copy2(src, dst)
    
    # Copy only the essential txt files from train folder
    essential_txt_files = ["train_A.txt", "train_B.txt", "train_gt.txt", "train_synthetic_A.txt"]
    for txt_name in essential_txt_files:
        txt_file = train_src / txt_name
        if txt_file.exists():
            print(f"  Copying {txt_name}...")
            shutil.copy2(txt_file, train_dst / txt_name)
    
    return temp_dir


def main():
    parser = argparse.ArgumentParser(description="Upload synthetic dataset to Hugging Face Hub")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the dataset folder")
    parser.add_argument("--repo_name", type=str, required=True,
                       help="Hugging Face repo name (e.g., 'username/dataset-name')")
    parser.add_argument("--dataset_name", type=str, default="Synthetic Change Detection Dataset",
                       help="Human-readable dataset name")
    parser.add_argument("--original_dataset_url", type=str, required=True,
                       help="URL to the original dataset")
    parser.add_argument("--original_dataset_name", type=str, default=None,
                       help="Name of the original dataset (e.g., 'SECOND', 'CNAM-CD')")
    parser.add_argument("--private", action="store_true",
                       help="Make the dataset private")
    parser.add_argument("--class_names", type=str, nargs="+",
                       default=["non-change", "impervious surface", "bare ground", 
                               "low vegetation", "medium vegetation", "non-vegetated ground surface",
                               "tree", "water bodies", "building", "playground"],
                       help="List of class names (unified 10-class taxonomy)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Prepare files but don't upload")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    # Infer original dataset name if not provided
    original_dataset_name = args.original_dataset_name
    if original_dataset_name is None:
        if "second" in str(dataset_path).lower():
            original_dataset_name = "SECOND"
        elif "cnam" in str(dataset_path).lower():
            original_dataset_name = "CNAM-CD"
        else:
            original_dataset_name = "Original"

    # Count samples
    num_b, num_synthetic_a = count_samples(dataset_path)
    print(f"Found {num_b} B samples, {num_synthetic_a} unique synthetic_A samples")

    if not args.dry_run:
        # Create the repository
        print(f"Creating repository: {args.repo_name}")
        api = HfApi()
        
        try:
            create_repo(
                repo_id=args.repo_name,
                repo_type="dataset",
                private=args.private,
                exist_ok=True
            )
        except Exception as e:
            print(f"Note: {e}")

    # Create temporary directory for upload (excluding A/)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print("Preparing upload folder (excluding A/ from original dataset)...")
        prepare_upload_folder(dataset_path, temp_path)
        
        # Create README.md
        readme_content = create_dataset_card(
            args.dataset_name, 
            num_b,
            num_synthetic_a,
            args.class_names,
            args.original_dataset_url,
            original_dataset_name,
        )
        readme_path = temp_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"Created dataset card: {readme_path}")

        if args.dry_run:
            print(f"\n[DRY RUN] Would upload the following structure:")
            for item in sorted(temp_path.rglob("*")):
                if item.is_file():
                    rel_path = item.relative_to(temp_path)
                    size = item.stat().st_size / (1024 * 1024)  # MB
                    print(f"  {rel_path} ({size:.2f} MB)")
            print(f"\n[DRY RUN] Total files would be uploaded to: {args.repo_name}")
            return

        # Upload the folder
        print(f"\nUploading dataset to {args.repo_name}...")
        print("This may take a while for large datasets...")
        
        upload_folder(
            folder_path=str(temp_path),
            repo_id=args.repo_name,
            repo_type="dataset",
            commit_message=f"Upload {args.dataset_name}",
        )
    
    print(f"\n✅ Dataset uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{args.repo_name}")
    print(f"\n⚠️  Remember: Users need to download the original {original_dataset_name} dataset")
    print(f"   for the A/ images from: {args.original_dataset_url}")


if __name__ == "__main__":
    main()
