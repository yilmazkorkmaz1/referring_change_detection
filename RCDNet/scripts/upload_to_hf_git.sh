#!/bin/bash
# Upload synthetic datasets to Hugging Face using hf upload
# This handles large files better than the Python API

set -e

# Configuration
SECOND_DATASET_PATH="/mnt/store/ykorkma1/datasets/second_dataset_with_1ch"
CNAMCD_DATASET_PATH="/mnt/store/ykorkma1/datasets/cnamcd_dataset_with_1ch"
SECOND_REPO="yilmazkorkmaz/Synthetic_RCD_1"
CNAMCD_REPO="yilmazkorkmaz/Synthetic_RCD_2"
WORK_DIR="/mnt/store/ykorkma1/hf_uploads"

# Create README content
create_readme() {
    local dataset_name="$1"
    local num_samples="$2"
    local num_synthetic_a="$3"
    local original_dataset_name="$4"
    local original_dataset_url="$5"
    local use_zip="${6:-true}"
    
    cat << 'HEREDOC_END'
---
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

HEREDOC_END

    cat << EOF
# ${dataset_name}

This is a **synthetic** change detection dataset for training [RCDNet](https://github.com/yilmazkorkmaz/RCDNet).

## Important: Pre-change Images (A/)

**This dataset does NOT include the pre-change images (A/)** because they come from the original ${original_dataset_name} dataset. You have two options:

### Option 1: Use Original Dataset (Recommended)

1. Download the original ${original_dataset_name} dataset from: ${original_dataset_url}
2. Create a symlink to the A/ folder:

\`\`\`bash
ln -s /path/to/original/${original_dataset_name,,}/train/A train/A
\`\`\`

### Option 2: Use Synthetic A from B/ (No Original Dataset Needed)

The \`train_synthetic_A.txt\` file maps each sample to a "no-change" image already in B/ (every 10th: 0.png, 10.png, 20.png...). These ${num_synthetic_a} unique images were generated with "no-change" prompts and can serve as A images:

\`\`\`bash
python scripts/download_from_hf.py \\
  --repo_name yilmazkorkmaz/Synthetic_RCD_1 \\
  --local_dir data/second_synthetic \\
  --use_synthetic_a
\`\`\`

## Dataset Description

- **Synthetic B samples**: ${num_samples}
- **Synthetic A samples**: ${num_synthetic_a} (no-change images in B/, every 10th)
- **Image Size**: 512x512
- **Format**: PNG images
- **Classes**: 10

### What's Included

| File | Description |
|------|-------------|
| \`train/B-{00000..00009}.tar\` | Sharded synthetic post-change images (${num_samples} samples) |
| \`train/gt-{00000..00009}.tar\` | Sharded ground truth semantic change masks |
| \`train/train_A.txt\` | File list mapping to original dataset A/ images |
| \`train/train_B.txt\` | File list for synthetic B/ images |
| \`train/train_gt.txt\` | File list for ground truth masks |
| \`train/train_synthetic_A.txt\` | File list for no-change samples (every 10th in B/) |

### Download and Extract

\`\`\`bash
# Download the dataset
huggingface-cli download yilmazkorkmaz/Synthetic_RCD_1 --repo-type dataset --local-dir data/second_synthetic

# Extract all sharded tar files
cd data/second_synthetic/train
mkdir -p B gt
for f in B-*.tar; do tar -xf "\$f" -C B/; done
for f in gt-*.tar; do tar -xf "\$f" -C gt/; done

# Link A/ from original dataset (Option 1)
ln -s /path/to/second_dataset/train/A A/
\`\`\`

### Load with HuggingFace Datasets (Streaming)

You can stream the images directly without downloading the full dataset:

\`\`\`python
from datasets import load_dataset

# Stream B images from all shards
dataset = load_dataset(
    "webdataset",
    data_files="https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_1/resolve/main/train/B-*.tar",
    split="train",
    streaming=True
)

for sample in dataset:
    print(sample)  # {'__key__': '0', '__url__': '...', 'png': <image bytes>}
\`\`\`

You can also load specific shards for parallel processing:

\`\`\`python
# Load only first 2 shards
dataset = load_dataset(
    "webdataset",
    data_files=[
        "https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_1/resolve/main/train/B-00000.tar",
        "https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_1/resolve/main/train/B-00001.tar",
    ],
    split="train",
    streaming=True
)
\`\`\`

### Class Names

\`\`\`
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
\`\`\`

## Image Enumeration Scheme

The synthetic images follow a specific naming convention that encodes both the pre-change image (A) and the change class:

\`\`\`
image_index = (A_index * 10) + class_index
\`\`\`

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
- \`class_index = image_index % 10\` → determines which class changed
- \`A_index = image_index // 10\` → determines which pre-change image was used
- Images where \`image_index % 10 == 0\` are "no-change" images (synthetic_A)

## Citation

If you use this dataset, please cite both RCDNet and the original ${original_dataset_name} dataset.

\`\`\`bibtex
@inproceedings{korkmaz2026referring,
  title     = {Referring Change Detection in Remote Sensing Imagery},
  author    = {Korkmaz, Yilmaz and Paranjape, Jay N. and de Melo, Celso M. and Patel, Vishal M.},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
\`\`\`

## License

This synthetic data is released under the MIT License. Please also respect the license of the original ${original_dataset_name} dataset.
EOF
}

upload_dataset() {
    local dataset_path="$1"
    local repo_name="$2"
    local dataset_name="$3"
    local original_dataset_name="$4"
    local original_dataset_url="$5"
    local use_zip="${6:-true}"  # Default to using zip
    
    echo "=============================================="
    echo "Uploading: ${dataset_name}"
    echo "From: ${dataset_path}"
    echo "To: ${repo_name}"
    echo "Using ZIP: ${use_zip}"
    echo "=============================================="
    
    # Create work directory
    local upload_dir="${WORK_DIR}/$(basename ${repo_name})"
    
    # Count samples
    local num_b=$(sort -u "${dataset_path}/train/train_B.txt" | wc -l)
    local num_synthetic_a=$(sort -u "${dataset_path}/train/train_synthetic_A.txt" | wc -l)
    
    if [ "${use_zip}" = "true" ]; then
        # ZIP mode - create zip files instead of copying individual files
        rm -rf "${upload_dir}"
        mkdir -p "${upload_dir}/train"
        
        # Check if sharded tar files already exist
        if ls "${upload_dir}/train/B-"*.tar 1>/dev/null 2>&1; then
            echo "Sharded TAR files already exist, skipping compression..."
        else
            # Create sharded tar files (~3000 images per shard)
            local shard_size=3000
            local files_list=$(sort -u "${dataset_path}/train/train_B.txt")
            local total_files=$(echo "$files_list" | wc -l)
            local num_shards=$(( (total_files + shard_size - 1) / shard_size ))
            
            echo "Creating ${num_shards} sharded TAR files (${shard_size} images each)..."
            
            # Create B shards
            cd "${dataset_path}/train/B"
            local shard_idx=0
            echo "$files_list" | split -l ${shard_size} -d -a 5 - /tmp/shard_
            for shard_file in /tmp/shard_*; do
                local shard_name=$(printf "B-%05d.tar" $shard_idx)
                echo "  Creating ${shard_name}..."
                tar -cf "${upload_dir}/train/${shard_name}" $(cat "$shard_file")
                shard_idx=$((shard_idx + 1))
                rm "$shard_file"
            done
            
            # Create gt shards
            local gt_files_list=$(sort -u "${dataset_path}/train/train_gt.txt")
            cd "${dataset_path}/train/gt"
            shard_idx=0
            echo "$gt_files_list" | split -l ${shard_size} -d -a 5 - /tmp/shard_
            for shard_file in /tmp/shard_*; do
                local shard_name=$(printf "gt-%05d.tar" $shard_idx)
                echo "  Creating ${shard_name}..."
                tar -cf "${upload_dir}/train/${shard_name}" $(cat "$shard_file")
                shard_idx=$((shard_idx + 1))
                rm "$shard_file"
            done
        fi
        
        echo "Copying txt files..."
        cp "${dataset_path}/train/train_A.txt" "${upload_dir}/train/"
        cp "${dataset_path}/train/train_B.txt" "${upload_dir}/train/"
        cp "${dataset_path}/train/train_gt.txt" "${upload_dir}/train/"
        cp "${dataset_path}/train/train_synthetic_A.txt" "${upload_dir}/train/"
        
    else
        # Non-zip mode - copy individual files
        if [ -d "${upload_dir}/train/B" ] && [ "$(ls -A ${upload_dir}/train/B 2>/dev/null | head -1)" ]; then
            echo "Staging folder already exists with files, skipping copy..."
            echo "Found $(ls ${upload_dir}/train/B | wc -l) B images in staging folder"
        else
            rm -rf "${upload_dir}"
            mkdir -p "${upload_dir}/train/B" "${upload_dir}/train/gt"
            
            echo "Copying ${num_b} B images..."
            for f in $(sort -u "${dataset_path}/train/train_B.txt"); do
                cp "${dataset_path}/train/B/${f}" "${upload_dir}/train/B/" 2>/dev/null || true
            done
            
            echo "Copying ${num_b} gt images..."
            for f in $(sort -u "${dataset_path}/train/train_gt.txt"); do
                cp "${dataset_path}/train/gt/${f}" "${upload_dir}/train/gt/" 2>/dev/null || true
            done
            
            echo "Copying txt files..."
            cp "${dataset_path}/train/train_A.txt" "${upload_dir}/train/"
            cp "${dataset_path}/train/train_B.txt" "${upload_dir}/train/"
            cp "${dataset_path}/train/train_gt.txt" "${upload_dir}/train/"
            cp "${dataset_path}/train/train_synthetic_A.txt" "${upload_dir}/train/"
        fi
    fi
    
    echo "Creating README.md..."
    create_readme "${dataset_name}" "${num_b}" "${num_synthetic_a}" "${original_dataset_name}" "${original_dataset_url}" "${use_zip}" > "${upload_dir}/README.md"
    
    echo "Uploading to Hugging Face..."
    cd "${upload_dir}"
    
    # For zip mode, regular upload is fine (only a few files)
    if [ "${use_zip}" = "true" ]; then
        huggingface-cli upload "${repo_name}" . --repo-type=dataset
    else
        # For non-zip mode, use large-folder with retry
        max_retries=10
        retry_count=0
        wait_time=300
        
        while [ $retry_count -lt $max_retries ]; do
            if huggingface-cli upload-large-folder "${repo_name}" . --repo-type=dataset; then
                break
            else
                retry_count=$((retry_count + 1))
                if [ $retry_count -lt $max_retries ]; then
                    echo "Rate limited - waiting ${wait_time}s (attempt ${retry_count}/${max_retries})..."
                    sleep ${wait_time}
                else
                    echo "Max retries reached."
                    return 1
                fi
            fi
        done
    fi
    
    echo "=============================================="
    echo "Done! View at: https://huggingface.co/datasets/${repo_name}"
    echo "=============================================="
}

# Main
case "${1:-}" in
    "second")
        upload_dataset \
            "${SECOND_DATASET_PATH}" \
            "${SECOND_REPO}" \
            "Synthetic RCD Dataset 1 (SECOND)" \
            "SECOND" \
            "https://captain-whu.github.io/SCD/"
        ;;
    "cnamcd")
        upload_dataset \
            "${CNAMCD_DATASET_PATH}" \
            "${CNAMCD_REPO}" \
            "Synthetic RCD Dataset 2 (CNAM-CD)" \
            "CNAM-CD" \
            "https://github.com/Chen-Yang-Liu/CNAM-CD"
        ;;
    "all")
        $0 second
        $0 cnamcd
        ;;
    *)
        echo "Usage: $0 {second|cnamcd|all}"
        echo ""
        echo "  second  - Upload SECOND synthetic dataset to ${SECOND_REPO}"
        echo "  cnamcd  - Upload CNAM-CD synthetic dataset to ${CNAMCD_REPO}"
        echo "  all     - Upload both datasets"
        exit 1
        ;;
esac
