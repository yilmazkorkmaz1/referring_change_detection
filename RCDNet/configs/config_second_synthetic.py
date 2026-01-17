import os
import numpy as np
from easydict import EasyDict as edict

# Config for training on synthetic second_dataset_synthetic
# but validating on original second_dataset
C = edict()
config = C
cfg = C

C.seed = 40

# -------------------------
# Dataset
# -------------------------
C.dataset_name = "second_dataset_synthetic"

# Train on synthetic data, validate/test on original real data
C.root_by_split = {
    "train": os.path.abspath(os.path.join(os.getcwd(), "data", "second_dataset_synthetic")),
    "val": os.path.abspath(os.path.join(os.getcwd(), "data", "second_dataset")),
    "test": os.path.abspath(os.path.join(os.getcwd(), "data", "second_dataset")),
}
# Fallback (used if root_by_split doesn't have the split)
C.root_folder = os.path.abspath(os.path.join(os.getcwd(), "data", "second_dataset_synthetic"))

C.A_format = ".png"
C.B_format = ".png"
C.gt_format = ".png"

# 10-class taxonomy for synthetic training data
C.class_names = [
    "non-change",
    "impervious surface",
    "bare ground",
    "low vegetation",
    "medium vegetation",
    "non-vegetated ground surface",
    "tree",
    "water bodies",
    "building",
    "playground",
]

# 7-class taxonomy for original validation/test data
C.class_names_by_split = {
    "val": [
        "Non-change",
        "Low Vegetation",
        "Non-vegetated Ground Surface",
        "Tree",
        "Water",
        "Building",
        "Playground",
    ],
    "test": [
        "Non-change",
        "Low Vegetation",
        "Non-vegetated Ground Surface",
        "Tree",
        "Water",
        "Building",
        "Playground",
    ],
}

# Use validation num_classes for evaluation (7 for original dataset)
C.num_classes = 7

C.train_split = "train"
C.val_split = "val"
C.test_split = "test"
C.eval_class_selection = "first"

# Images
C.image_height = 512
C.image_width = 512

# -------------------------
# Model
# -------------------------
C.backbone = "sigma_small"  # sigma_tiny / sigma_small / sigma_base
C.decoder = "MambaDecoder"
C.decoder_embed_dim = 512
C.use_imagenet_pretrain = True
C.freeze_backbone = False
C.pretrained_model = None
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# -------------------------
# Train
# -------------------------
C.lr = 6e-5
C.weight_decay = 0.01
C.batch_size = 4
C.nepochs = 100
C.num_workers = 8

# Augmentation
C.use_color_jitter = True
C.jitter_hyper = 0.1

# Dataset-specific normalization statistics (synthetic)
# Averaged for single normalization
C.norm_mean = np.array([0.436, 0.429, 0.408])
C.norm_std = np.array([0.223, 0.205, 0.213])
C.use_cached_norm = False
C.use_single_normalization = True

# -------------------------
# Naming
# -------------------------
C.trial_name = f"{C.dataset_name}_{C.backbone}_{C.decoder}"
