import os
import numpy as np
from easydict import EasyDict as edict

# Minimal config for the open-source training/eval pipeline (train.py + eval.py)
C = edict()
config = C
cfg = C  # legacy alias

# Repro
C.seed = 3407

# Dataset
C.dataset_name = "second_dataset"
# Default assumes repo-local data layout; you can override via --data_root in train.py/eval.py
C.root_folder = os.path.abspath(os.path.join(os.getcwd(), "data", "second_dataset"))
C.A_format = ".png"
C.B_format = ".png"
C.gt_format = ".png"
C.num_classes = 7
C.class_names = [
    "Non-change",
    "Low Vegetation",
    "Non-vegetated Ground Surface",
    "Tree",
    "Water",
    "Building",
    "Playground",
]
C.train_split = "train"
C.val_split = "val"
C.test_split = "test"
C.eval_class_selection = "first"

# Images
C.image_height = 512
C.image_width = 512

# Model
C.backbone = "sigma_small"  # sigma_tiny / sigma_small / sigma_base
C.decoder = "MambaDecoder"  # MambaDecoder / MLPDecoder / ...
C.decoder_embed_dim = 512
C.pretrained_model = None
C.freeze_backbone = False
C.use_imagenet_pretrain = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# Train
C.lr = 6e-5
C.weight_decay = 0.01
C.batch_size = 2
C.nepochs = 500
C.num_workers = 16

# Augmentation (ChangeDataset)
C.use_color_jitter = False
C.jitter_hyper = 0.1

# Dataset-specific normalization statistics (averaged A and B)
# A: [0.441, 0.444, 0.452], B: [0.437, 0.450, 0.466]
C.norm_mean = np.array([0.439, 0.447, 0.459])
C.norm_std = np.array([0.193, 0.183, 0.189])
C.use_cached_norm = False
C.use_single_normalization = True

# Naming (used by train.py output_dir)
C.trial_name = f"{C.dataset_name}_{C.backbone}_{C.decoder}"