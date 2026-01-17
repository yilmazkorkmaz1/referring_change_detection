import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

# Mixed synthetic dataset training config
# Trains on second_dataset_synthetic + cnamcd_dataset_synthetic with unified taxonomy

C = edict()
config = C
cfg = C

C.seed = 40
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), "./"))

# -------------------------
# Dataset (mixed synthetic)
# -------------------------
C.dataset_name = "SYNTHETIC_MIXED_SECOND_CNAMCD"

# Unified 10-class taxonomy for mixed training
# All datasets are mapped to this common class space
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

# Class mappings (original dataset index -> unified index)
# SECOND original: [Non-change, Low Vegetation, Non-vegetated Ground Surface, Tree, Water, Building, Playground]
second_class_mapping = {0: 0, 1: 3, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9}
# CNAM-CD original: [background, Impervious surface, bare ground, vegetation, Water bodies, Other]
cnamcd_class_mapping = {0: 0, 1: 1, 2: 2, 3: 4, 4: 7, 5: 2}

# Mixed training: use plain list to avoid EasyDict int-key issues
# Train on synthetic datasets, validate/test on real datasets
root_folder_list = [
    {
        # SECOND: synthetic for train, real for val/test
        "root_by_split": {
            "train": osp.abspath(osp.join(C.root_dir, "data", "second_dataset_synthetic")),
            "val": osp.abspath(osp.join(C.root_dir, "data", "second_dataset")),
            "test": osp.abspath(osp.join(C.root_dir, "data", "second_dataset")),
        },
        "A_format": ".png",
        "B_format": ".png",
        "gt_format": ".png",
        "class_names": unified_class_names,
        # SECOND synthetic: B_mean + synthetic_A_mean averaged
        # B: [0.467, 0.464, 0.460], synth_A: [0.404, 0.394, 0.356]
        "norm_mean": np.array([0.436, 0.429, 0.408]),
        "norm_std": np.array([0.223, 0.205, 0.213]),
        "image_size": (512, 512),
        "class_mapping": second_class_mapping,
    },
    {
        # CNAM-CD: synthetic for train, real for val/test
        "root_by_split": {
            "train": osp.abspath(osp.join(C.root_dir, "data", "cnamcd_dataset_synthetic")),
            "val": osp.abspath(osp.join(C.root_dir, "data", "CNAM-CD")),
            "test": osp.abspath(osp.join(C.root_dir, "data", "CNAM-CD")),
        },
        "A_format": ".png",  # synthetic uses .png
        "B_format": ".png",
        "gt_format": ".png",
        # Real CNAM-CD uses .tif - handled per-split below
        "A_format_by_split": {"train": ".png", "val": ".tif", "test": ".tif"},
        "B_format_by_split": {"train": ".png", "val": ".tif", "test": ".tif"},
        "gt_format_by_split": {"train": ".png", "val": ".tif", "test": ".tif"},
        "class_names": unified_class_names,
        # CNAM-CD synthetic: B_mean + synthetic_A_mean averaged
        # B: [0.464, 0.448, 0.418], synth_A: [0.396, 0.372, 0.308]
        "norm_mean": np.array([0.430, 0.410, 0.363]),
        "norm_std": np.array([0.244, 0.194, 0.195]),
        "image_size": (512, 512),
        "class_mapping": cnamcd_class_mapping,
    },
]
object.__setattr__(C, 'root_folder', root_folder_list)

# Defaults used by code paths that expect single-dataset fields
C.A_format = ".png"
C.B_format = ".png"
C.gt_format = ".png"
C.class_names = unified_class_names  # Use unified taxonomy
C.unified_class_names = unified_class_names  # Used by dataloader for ConcatDataset
C.image_height = 512
C.image_width = 512
C.num_classes = 10  # Unified taxonomy has 10 classes

C.train_split = "train"
C.val_split = "val"
C.test_split = "test"
C.eval_class_selection = "first"

# Normalization settings
# Note: Each dataset in root_folder_list has its own norm_mean/norm_std
# SECOND synthetic uses: [0.436, 0.429, 0.408] mean, [0.223, 0.205, 0.213] std
# CNAM-CD synthetic uses: [0.430, 0.410, 0.363] mean, [0.244, 0.194, 0.195] std
C.use_cached_norm = False
C.use_single_normalization = True  # Each dataset uses single norm for both A and B

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

# Pretrained checkpoint to initialize model weights (fine-tuning)
# Set to None to train from scratch, or path to .pt/.safetensors checkpoint
C.pretrained_checkpoint = None

# -------------------------
# Train
# -------------------------
C.lr = 6e-5
C.weight_decay = 0.01
C.batch_size = 4
C.nepochs = 100
C.num_workers = 8

# Dataset augmentation (applies to all mixed datasets)
C.use_color_jitter = False
C.jitter_hyper = 0.1

# -------------------------
# Logging
# -------------------------
C.trial_name = f"{C.dataset_name}_{C.backbone}_{C.decoder}"
C.log_dir = osp.abspath(osp.join(C.root_dir, "runs", C.trial_name))
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoints"))

