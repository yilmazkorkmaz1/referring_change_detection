# RCDNet: Referring Change Detection Network

**RCDNet** is a language-guided referring change detection model for remote sensing imagery. Given a *before* image (A), an *after* image (B), and a **text prompt** (e.g., "building", "water bodies"), it predicts **which pixels of that semantic class have changed**. By querying all classes sequentially, RCDNet produces dense **semantic change masks** that capture fine-grained land cover transitions.


## Installation

```bash
# Option 1: Conda (recommended)
conda env create -f environment.yml
conda activate rcdnet

# Build Selective Scan (required)
pip install -e models/encoders/selective_scan
```

Download VMamba pretrained weights from [Sigma GitHub](https://github.com/zifuwan/Sigma) and place them in `pretrained/vmamba/`:
- `vssmsmall_dp03_ckpt_epoch_238.pth` (default) - [Download](https://github.com/zifuwan/Sigma/blob/main/pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth)
- `vssmtiny_dp01_ckpt_epoch_292.pth` - [Download](https://github.com/zifuwan/Sigma/blob/main/pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth)
- `vssmbase_dp06_ckpt_epoch_241.pth` - [Download](https://github.com/zifuwan/Sigma/blob/main/pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth)

## Datasets

Synthetic datasets are **optional**. You can train and evaluate RCDNet using only real datasets (e.g., SECOND, CNAM-CD) without downloading or using any synthetic data. Synthetic data is provided only as an optional add-on for synthetic pre-training / augmentation experiments.

### (Optional) Download Synthetic Datasets from Hugging Face

We provide synthetic training datasets on Hugging Face. These contain synthetic B/ (post-change) images and gt/ masks. For pre-change images (A/), you have two options:

```bash
pip install huggingface_hub

# Option 1: Link to original dataset's A/ folder (recommended)
python scripts/download_from_hf.py \
  --repo_name yilmazkorkmaz/Synthetic_RCD_1 \
  --local_dir data/second_synthetic \
  --original_a_path /path/to/second_dataset/train/A

# Option 2: Use synthetic_A from B/ (no-change images, every 10th sample)
python scripts/download_from_hf.py \
  --repo_name yilmazkorkmaz/Synthetic_RCD_1 \
  --local_dir data/second_synthetic \
  --use_synthetic_a
```

| Dataset | Hugging Face Repo | Original Dataset |
|---------|-------------------|------------------|
| SECOND Synthetic | [`yilmazkorkmaz/Synthetic_RCD_1`](https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_1) | [SECOND](https://captain-whu.github.io/SCD/) |
| CNAM-CD Synthetic | [`yilmazkorkmaz/Synthetic_RCD_2`](https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_2) | [CNAM-CD](https://github.com/Chen-Yang-Liu/CNAM-CD) |

**Synthetic image naming:** `image_index = (A_index × 10) + class_index`. Each group of 10 images shares the same pre-change image, with each representing a different change class (0=non-change, 1=impervious surface, ..., 9=playground). See [docs/DATASETS.md](docs/DATASETS.md) for details.

### Dataset Format

```
dataset_root/
├── A/              # Pre-change images
├── B/              # Post-change images
├── gt/             # Ground truth masks (class indices: 0, 1, 2, ...)
├── train.txt       # Sample IDs (one per line, no extension)
├── val.txt
└── test.txt
```

Ground truth masks use **class indices** (0=background, 1=class1, 2=class2, ...), NOT binary (0/255).

**Note:** All normalization statistics are pre-computed and hardcoded in the config files. No external JSON files are needed.

See [docs/DATASETS.md](docs/DATASETS.md) for detailed format guide.

## Training

### Single Dataset

```bash
# SECOND dataset
accelerate launch train.py \
  --config configs.config_second \
  --data_root /path/to/second_dataset \
  --wandb --amp

# CNAM-CD dataset
accelerate launch train.py \
  --config configs.config_cnam \
  --data_root /path/to/cnam_cd \
  --wandb --amp
```

### Mixed Dataset (SECOND + CNAM-CD)

Create a mixed config or use command line:

```bash
accelerate launch train.py \
  --config configs.config_mixed \
  --wandb --amp
```

Example `configs/config_mixed.py` - uses unified 10-class taxonomy:
```python
# Unified taxonomy that both datasets map to
unified_class_names = [
    "non-change",                    # 0
    "impervious surface",            # 1 (CNAM-CD)
    "bare ground",                   # 2 (CNAM-CD)
    "low vegetation",                # 3 (SECOND)
    "medium vegetation",             # 4 (CNAM-CD "vegetation")
    "non-vegetated ground surface",  # 5 (SECOND)
    "tree",                          # 6 (SECOND)
    "water bodies",                  # 7 (SECOND + CNAM-CD)
    "building",                      # 8 (SECOND)
    "playground",                    # 9 (SECOND)
]

# Class mappings: original index -> unified index
second_class_mapping = {0: 0, 1: 3, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9}
cnamcd_class_mapping = {0: 0, 1: 1, 2: 2, 3: 4, 4: 7, 5: 2}

C.root_folder = [
    {
        "root": "/path/to/second_dataset",
        "class_names": unified_class_names,
        "class_mapping": second_class_mapping,
    },
    {
        "root": "/path/to/cnam_cd",
        "class_names": unified_class_names,
        "class_mapping": cnamcd_class_mapping,
    },
]
C.num_classes = 10  # Unified taxonomy
```

### Synthetic Pre-training

Optional: train on synthetic data, validate on real:

```bash
accelerate launch train.py \
  --config configs.config_second_synthetic \
  --wandb --amp
```

### Training Options

| Option | Description |
|--------|-------------|
| `--config` | Config module path (e.g., `configs.config_second`) |
| `--data_root` | Override dataset root |
| `--epochs` | Number of epochs |
| `--batch_size` | Batch size per GPU |
| `--lr` | Learning rate |
| `--amp` | Mixed precision training |
| `--wandb` | Enable W&B logging |
| `--eval_every` | Evaluate every N epochs |
| `--resume` | Resume from checkpoint |

## Evaluation

```bash
python eval.py \
  --config configs.config_second \
  --data_root /path/to/dataset \
  --split val \
  --checkpoint runs/.../checkpoints/best.pt
```

Supports `.pt`, `.pth`, and `.safetensors` checkpoint formats.

### Metrics

| Metric | Description |
|--------|-------------|
| **Score** | Primary metric (0.3 × Semantic_IoU + 0.7 × SeK) |
| **SeK** | Semantic change detection score (κ × exp(IoU_fg) / e) |
| **Semantic_IoU** | Mean IoU over semantic classes (foreground + background) |

## Configuration Reference

| Field | Description | Default |
|-------|-------------|---------|
| `root_folder` | Dataset root (str or list for mixed) | Required |
| `root_by_split` | Per-split dataset roots | None |
| `num_classes` | Number of classes (incl. background) | Required |
| `class_names` | List of class names | Required |
| `backbone` | `sigma_tiny`, `sigma_small`, `sigma_base` | `sigma_small` |
| `decoder` | `MambaDecoder`, `MLPDecoder` | `MambaDecoder` |
| `image_height/width` | Input image size | 512 |
| `lr` | Learning rate | 6e-5 |
| `batch_size` | Batch size | 2 x 8 (number of GPUs)|
| `nepochs` | Total epochs | 500 |


## Acknowledgments

This work builds upon several excellent open-source projects:

- **[VMamba](https://github.com/MzeroMiko/VMamba)** - Visual State Space Models providing the backbone architecture.
- **[Sigma](https://github.com/zifuwan/Sigma)** - Siamese Mamba Network for Multi-Modal Semantic Segmentation.
- **[M-CD](https://github.com/JayParanjape/M-CD)** - Mamba-based Change Detection for Remote Sensing.
- **[DDPM-CD](https://github.com/wgcban/ddpm-cd)** - Some of the data splits and preprocessing are adapted from this work on Remote Sensing Change Detection using Denoising Diffusion Probabilistic Models.

We thank the authors of these projects for their valuable contributions to the community.
