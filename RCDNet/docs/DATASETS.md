# Dataset Guide for RCDNet

This document explains the dataset formats, configurations, and how to set up training with different data sources.

## Table of Contents
- [Downloading Datasets](#downloading-datasets)
- [Dataset Structure](#dataset-structure)
- [Ground Truth Format](#ground-truth-format)
  - [Converting RGB Labels to Class Indices](#converting-rgb-labels-to-class-indices)
- [Configuration Options](#configuration-options)
- [Training Scenarios](#training-scenarios)
- [Normalization](#normalization)
- [Troubleshooting](#troubleshooting)

---

## Downloading Datasets

### Synthetic Datasets from Hugging Face

We provide synthetic training datasets on Hugging Face Hub. **Important:** These datasets contain only:
- **B/** - Synthetic post-change images (generated)
- **gt/** - Ground truth semantic change masks
- **train_synthetic_A.txt** - Maps each sample to a "no-change" image in B/ (every 10th)

The **A/** (pre-change images) come from the original datasets and are NOT included to avoid redistribution.

| Dataset | Hugging Face Repo | Original Dataset Required |
|---------|-------------------|---------------------------|
| SECOND Synthetic | [`yilmazkorkmaz/Synthetic_RCD_1`](https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_1) | [SECOND](https://captain-whu.github.io/SCD/) |
| CNAM-CD Synthetic | [`yilmazkorkmaz/Synthetic_RCD_2`](https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_2) | [CNAM-CD](https://github.com/Chen-Yang-Liu/CNAM-CD) |

#### Option 1: Use with Original Dataset (Recommended)

Download both the synthetic data and original dataset, then link them:

```bash
python scripts/download_from_hf.py \
  --repo_name yilmazkorkmaz/Synthetic_RCD_1 \
  --local_dir data/second_synthetic \
  --original_a_path /path/to/second_dataset/train/A

# This will:
# - Download B/, gt/ from Hugging Face
# - Create symlink: A/ -> /path/to/second_dataset/train/A
```

Or manually:
```bash
# Download
huggingface-cli download yilmazkorkmaz/Synthetic_RCD_1 --repo-type dataset --local-dir data/second_synthetic

# Link A/ folder from original dataset
ln -s /path/to/second_dataset/train/A data/second_synthetic/train/A
```

#### Option 2: Use Synthetic A from B/ (No Original Dataset Needed)

The `train_synthetic_A.txt` maps each sample to a "no-change" image already in B/ (every 10th: 0.png, 10.png, 20.png...). These were generated with "no-change" prompts:

```bash
python scripts/download_from_hf.py \
  --repo_name yilmazkorkmaz/Synthetic_RCD_1 \
  --local_dir data/second_synthetic \
  --use_synthetic_a

# This will:
# - Download B/, gt/ from Hugging Face
# - Create A/ folder with symlinks to referenced images in B/
# - Copy train_synthetic_A.txt to train_A.txt
```

Or manually:
```bash
# Download
huggingface-cli download yilmazkorkmaz/Synthetic_RCD_1 --repo-type dataset --local-dir data/second_synthetic

# Create A/ from synthetic_A images in B/
mkdir -p data/second_synthetic/train/A
for f in $(sort -u data/second_synthetic/train/train_synthetic_A.txt); do
  ln -s ../B/$f data/second_synthetic/train/A/$f
done
cp data/second_synthetic/train/train_synthetic_A.txt data/second_synthetic/train/train_A.txt
```

#### Download Programmatically

```python
from huggingface_hub import snapshot_download
import os

# Download synthetic dataset
local_path = snapshot_download(
    repo_id="yilmazkorkmaz/Synthetic_RCD_1",
    repo_type="dataset",
    local_dir="data/second_synthetic"
)

# Link to original dataset's A/ folder
os.symlink(
    "/path/to/second_dataset/train/A",
    f"{local_path}/train/A"
)
```

### Synthetic Dataset Image Enumeration

The synthetic B/ images follow a specific naming convention that encodes both the pre-change image (A) and the change class:

```
image_index = (A_index × 10) + class_index
```

**Classes (unified 10-class taxonomy):**
```
0: non-change
1: impervious surface
2: bare ground
3: low vegetation
4: medium vegetation
5: non-vegetated ground surface
6: tree
7: water bodies
8: building
9: playground
```

**Example enumeration:**

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
- Images where `image_index % 10 == 0` are "no-change" images (used as synthetic_A)

This means every 10 consecutive images share the same pre-change image (A), with each image representing a different change class.

### Real Datasets

For real change detection datasets, please refer to the original sources:

| Dataset | Source | Classes |
|---------|--------|---------|
| SECOND | [Official](https://captain-whu.github.io/SCD/) | 7 |
| CNAM-CD | [Official](https://github.com/Silvestezhou/CNAM-CD) | 6 |
| LEVIR-CD | [Official](https://justchenhao.github.io/LEVIR/) | 2 |
| WHU-CD | [Official](https://www.dropbox.com/scl/fi/ci39uxbzwmxr3asxs5h1s/WHU-CD-256.zip?rlkey=fl82nzgzsw0wc6gi7botjuapu&e=2&dl=0) | 2 |

---

## Dataset Structure

RCDNet expects datasets in one of the following layouts:

### Layout A: Flat Structure
```
dataset_root/
├── A/                    # Pre-change images
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── B/                    # Post-change images
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── gt/                   # Ground truth semantic masks
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── train.txt             # Training sample IDs (one per line, no extension)
├── val.txt               # Validation sample IDs
└── test.txt              # Test sample IDs
```

### Layout B: Split Subfolders
```
dataset_root/
├── train/
│   ├── A/
│   ├── B/
│   ├── gt/
│   └── train.txt
├── val/
│   ├── A/
│   ├── B/
│   ├── gt/
│   └── val.txt
└── test/
    ├── A/
    ├── B/
    ├── gt/
    └── test.txt
```

### Layout C: Shared Evaluation Folder
```
dataset_root/
├── train/
│   ├── A/, B/, gt/
│   └── train.txt
└── evaluation/           # Shared folder for val and test
    ├── A/, B/, gt/
    ├── val.txt
    └── test.txt
```

### Layout D: Legacy Synthetic (Paired Lists)
For synthetic datasets with different source images for A and B:
```
dataset_root/
├── A/, B/, gt/
├── train_A.txt           # List of A image IDs
├── train_B.txt           # List of B image IDs  
├── train_gt.txt          # List of GT image IDs
├── train_synthetic_A.txt # (Optional) Synthetic A images
├── val_A.txt, val_B.txt, val_gt.txt
└── test_A.txt, test_B.txt, test_gt.txt
```

---

## Ground Truth Format

Ground truth masks are **single-channel images** where each pixel value represents a **class index**:

| Pixel Value | Meaning |
|-------------|---------|
| 0 | No change (background) |
| 1 | Class 1 (e.g., "Low Vegetation") |
| 2 | Class 2 (e.g., "Building") |
| ... | ... |
| N | Class N |

### Example: SECOND Dataset (7 classes)
```
0 = Non-change
1 = Low Vegetation  
2 = Non-vegetated Ground Surface
3 = Tree
4 = Water
5 = Building
6 = Playground
```

### Example: Synthetic Dataset (10 classes)
```
0 = non-change
1 = impervious surface
2 = bare ground
3 = low vegetation
4 = medium vegetation
5 = non-vegetated ground surface
6 = tree
7 = water bodies
8 = building
9 = playground
```

**Important**: The class indices in the GT mask must match the order of `class_names` in your config.

### Converting RGB Labels to Class Indices

If your dataset has RGB-encoded labels (e.g., colored masks where each class has a specific RGB color), you can use the `color_to_class.py` utility to convert them to the required single-channel format:

```bash
# Use default paths (data/second_dataset/test/label2 -> data/second_dataset/test/gt)
python dataloader/color_to_class.py

# Or specify custom paths
python dataloader/color_to_class.py --input path/to/rgb_labels --output path/to/output_gt
```

**Color Mapping for SECOND Dataset:**
```python
(255, 255, 255) → 0  # Non-change (White)
(0, 128, 0)     → 1  # Low Vegetation (Dark Green)
(128, 128, 128) → 2  # Non-vegetated Ground Surface (Gray)
(0, 255, 0)     → 3  # Tree (Bright Green)
(0, 0, 255)     → 4  # Water (Blue)
(128, 0, 0)     → 5  # Building (Dark Red)
(255, 0, 0)     → 6  # Playground (Red)
```

To customize for your dataset, edit the `color_to_class` dictionary in `dataloader/color_to_class.py`.

The script will:
- Convert RGB images to single-channel grayscale (class indices)
- Warn you about unmapped colors (pixels that don't match any defined color)
- Process all `.png` and `.jpg` files in the input folder

---

## Configuration Options

### Basic Single Dataset Config
```python
C.root_folder = "/path/to/dataset"
C.num_classes = 7
C.class_names = ["Non-change", "Low Vegetation", ...]
```

### Per-Split Roots (Train on Synthetic, Validate on Real)
```python
C.root_by_split = {
    "train": "/path/to/synthetic_dataset",
    "val": "/path/to/real_dataset",
    "test": "/path/to/real_dataset",
}
C.root_folder = "/path/to/synthetic_dataset"  # Fallback
```

### Per-Split Class Names (Different Taxonomies)
```python
C.class_names = [...]  # Default (used for train)

C.class_names_by_split = {
    "val": ["Non-change", "Class1", ...],   # 7 classes
    "test": ["Non-change", "Class1", ...],  # 7 classes
}
```

### Image Formats
```python
C.A_format = ".png"   # Pre-change image format
C.B_format = ".png"   # Post-change image format
C.gt_format = ".png"  # Ground truth format
```

### Image Size
```python
C.image_height = 512
C.image_width = 512
```

---

## Training Scenarios

### Scenario 1: Train and Validate on Same Dataset
```python
# configs/config_second.py
C.root_folder = "/path/to/second_dataset"
C.num_classes = 7
C.class_names = ["Non-change", "Low Vegetation", ...]
```

```bash
accelerate launch train.py --config configs.config_second --wandb --amp
```

### Scenario 2: Train on Synthetic, Validate on Real
Use this when you have synthetic training data but want to evaluate on real data:

```python
# configs/config_second_synthetic.py
C.root_by_split = {
    "train": "/path/to/synthetic_data",  # 10-class synthetic
    "val": "/path/to/real_data",          # 7-class real
    "test": "/path/to/real_data",
}

C.class_names = [...]  # 10-class names for training

C.class_names_by_split = {
    "val": [...],   # 7-class names for validation
    "test": [...],
}

C.num_classes = 7  # Use validation num_classes for metrics
```

```bash
accelerate launch train.py --config configs.config_second_synthetic --wandb --amp
```

### Scenario 3: Mixed Dataset Training
Combine multiple datasets with different class vocabularies:

```python
C.root_folder = [
    {
        "root": "/path/to/dataset1",
        "class_names": ["bg", "class1", "class2"],
        "A_format": ".png",
    },
    {
        "root": "/path/to/dataset2", 
        "class_names": ["bg", "classA", "classB", "classC"],
        "A_format": ".tif",
    },
]
```

---

## Normalization

### Option 1: ImageNet Normalization (Default)
If no normalization is specified, ImageNet statistics are used:
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### Option 2: Cached Per-Dataset Normalization
For best results, compute dataset-specific statistics and save to JSON:

```
dataset_root/train/train_norm_values.json
```

```json
{
    "A_mean": [0.412, 0.398, 0.371],
    "A_std": [0.213, 0.207, 0.199],
    "B_mean": [0.423, 0.405, 0.382],
    "B_std": [0.218, 0.211, 0.204]
}
```

Enable in config:
```python
C.use_cached_norm = True
C.use_single_normalization = False  # True = average A and B stats
```

**Note**: If `val_norm_values.json` or `test_norm_values.json` is missing, the loader automatically falls back to `train_norm_values.json`.

---

## Quick Reference

| Config Field | Description |
|--------------|-------------|
| `root_folder` | Dataset root path (string or list) |
| `root_by_split` | Per-split root paths (dict) |
| `class_names` | List of class names matching GT indices |
| `class_names_by_split` | Per-split class names (dict) |
| `num_classes` | Number of classes (including background) |
| `A_format`, `B_format`, `gt_format` | File extensions |
| `image_height`, `image_width` | Target image size |
| `use_cached_norm` | Load normalization from JSON |
| `use_single_normalization` | Average A/B normalization stats |

---

## Troubleshooting

### "FileNotFoundError: train.txt not found"
- Check your dataset layout matches one of the supported formats
- Verify the split files exist (`train.txt`, `val.txt`, `test.txt`)

### "Class index out of range"
- Ensure GT mask values match `num_classes` (0 to num_classes-1)
- Verify `class_names` list length equals `num_classes`

### "Shape mismatch during evaluation"
- GT masks should be single-channel (grayscale)
- Pixel values should be integers (0, 1, 2, ...), not RGB or binary 255
- If you have RGB-encoded labels, use `dataloader/color_to_class.py` to convert them

### "Wrong predictions / Low scores"
- Verify GT masks are class indices (0, 1, 2, ...), not RGB colors
- Check that class indices match the order in `class_names`
- Use `color_to_class.py` if your labels are RGB-encoded
