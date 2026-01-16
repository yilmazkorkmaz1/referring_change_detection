# Referring Change Detection Synthetic Data Generator (RCDGen)

Diffusion-based referring change detection synthetic data generator using InstructPix2Pix. Generates post-images in a given category along with the change mask.

## Installation

```bash
git clone git@github.com:yilmazkorkmaz1/referring_change_detection.git
cd referring_change_detection/RCDGen
conda env create -f environment.yml
conda activate rcdgen
```

### Custom Diffusers Pipeline

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout v0.31.0
pip install -e .
cd ..

cp RCDGenSDPipeline.py diffusers/src/diffusers/pipelines/stable_diffusion/
```

## Dataset Structure

```
dataset_root/
├── A/
│   ├── image1.png
│   └── image2.png
├── B/
│   ├── image1.png
│   └── image2.png
├── gt/
│   ├── image1.png
│   └── image2.png
├── train.txt
├── val.txt
└── test.txt
```

## Training

```bash
accelerate launch --mixed_precision="fp16" RCDGen_train.py \
    --pretrained_model_name_or_path=timbrooks/instruct-pix2pix \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=1 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --val_image_url="example.png"  \
    --validation_prompt="change in building" \
    --seed=42 \
    --report_to=wandb \
    --use_ema
```

Edit `RCDGen_train.py` to configure datasets:

```python
dataset_1 = remote.ChangeDataset("train", ".tif", ".tif", ".tif", 
    "/path/to/dataset1", 
    ['non-change', 'class1', 'class2'], 
    tokenizer=tokenizer)

dataset = ConcatDataset([dataset_1, dataset_2])
```

## Inference

```python
import torch
from PIL import Image
from diffusers.pipelines.stable_diffusion.RCDGenSDPipeline import StableDiffusionInstructPix2PixPipeline

model_id = "path/to/checkpoint"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="ema"
).to("cuda")

image = Image.open("before.png").convert("RGB")
images = pipe(
    "change in building",
    image=image,
    num_inference_steps=100,
    image_guidance_scale=1.5,
    guidance_scale=7.0,
).images

edited_image = images[0][0]
change_mask = images[1][0]
```

Or see `RCDGen_inference.py` for a complete inference example.

Codes are mainly adapted from https://github.com/huggingface/diffusers/tree/main/examples/instruct_pix2pix
