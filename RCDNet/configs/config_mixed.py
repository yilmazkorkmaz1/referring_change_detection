import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 40
remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'MIXED_SECOND_CNAMCD'

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
root_folder_list = [
    {
        "root": osp.abspath(osp.join(C.root_dir, "data", "second_dataset")),
        "A_format": ".png",
        "B_format": ".png",
        "gt_format": ".png",
        "class_names": unified_class_names,
        # SECOND dataset normalization (averaged A and B)
        "norm_mean": np.array([0.439, 0.447, 0.459]),
        "norm_std": np.array([0.193, 0.183, 0.189]),
        "image_size": (512, 512),
        "class_mapping": second_class_mapping,
    },
    {
        "root": osp.abspath(osp.join(C.root_dir, "data", "CNAM-CD")),
        "A_format": ".tif",
        "B_format": ".tif",
        "gt_format": ".tif",
        "class_names": unified_class_names,
        # CNAM-CD dataset normalization (averaged A and B)
        "norm_mean": np.array([0.394, 0.390, 0.358]),
        "norm_std": np.array([0.212, 0.180, 0.175]),
        "image_size": (512, 512),
        "class_mapping": cnamcd_class_mapping,
    },
]
object.__setattr__(C, 'root_folder', root_folder_list)

C.A_format = ".png"
C.B_format = ".png"
C.gt_format = ".png"
C.class_names = unified_class_names  # Use unified taxonomy
C.image_height = 512
C.image_width = 512
C.is_test = False
C.num_eval_imgs = 500
C.num_classes = 10  # Unified taxonomy has 10 classes

"""Image Config"""
#C.background = 256
C.image_height = 512
C.image_width = 512
# Default normalization (fallback - each dataset in root_folder_list has its own normalization)
# SECOND uses: [0.439, 0.447, 0.459] mean, [0.193, 0.183, 0.189] std
# CNAM-CD uses: [0.394, 0.390, 0.358] mean, [0.212, 0.180, 0.175] std
C.norm_mean = np.array([0.416, 0.418, 0.408])
C.norm_std = np.array([0.202, 0.181, 0.182])
C.use_cached_norm = False
C.use_single_normalization = True  # Each dataset uses single norm for both A and B

""" Settings for network, this would be different for each kind of model"""

C.backbone = 'sigma_small' # sigma_tiny / sigma_small / sigma_base
C.decoder = 'MambaDecoder' # 'MambaDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

# Pretrained checkpoint to initialize model weights (fine-tuning)
# Set to None to train from scratch, or path to .pt/.safetensors checkpoint
C.pretrained_checkpoint = None  # e.g., "runs/synthetic/checkpoints/best.pt"   

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 4
C.nepochs = 100
C.num_train_imgs = 87000
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 16
C.train_scale_array = [1]
C.train_scale_array = None
C.warm_up_epoch = 5

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
# C.eval_iter = 1
C.eval_stride_rate = 1
C.eval_scale_array = [1] 
C.eval_flip = False
C.eval_crop_size = [512, 512]

"""Store Config"""
C.checkpoint_start_epoch = 1
C.checkpoint_step = 1

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.use_color_jitter = False
C.jitter_hyper = 0.1

C.freeze_backbone = False
C.use_imagenet_pretrain = True
C.pretrained_model = None  # VMamba loads its own pretrained weights

# Use a single validation dataset by pointing `root_folder` to a single dataset root
# and evaluating with `eval.py --config configs.config_second` or `configs.config_cnamcd`.



C.trial_name = f"{C.dataset_name}_{C.backbone}_{C.decoder}"
C.log_dir = osp.abspath(osp.join(C.root_dir, "runs", C.trial_name))
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))


if C.use_imagenet_pretrain:
    if C.backbone == 'swin_b':
        C.pretrained_model = "swin_base_patch4_window7_224_22k.pth"
    elif C.backbone == 'swin_s':
        C.pretrained_model = "swin_small_patch4_window7_224_22k.pth"
    elif C.backbone == 'swin_l':
        C.pretrained_model = "swin_large_patch4_window12_384_22k.pth"
    elif C.backbone == 'mit_b5':
        C.pretrained_model = "pretrained/segformer/mit_b5.pth"
    elif C.backbone == 'mit_b4':
        C.pretrained_model = "pretrained/segformer/mit_b4.pth"
    elif C.backbone == 'mit_b3':
        C.pretrained_model = "pretrained/segformer/mit_b3.pth"
    elif C.backbone == 'mit_b2':
        C.pretrained_model = "pretrained/segformer/mit_b2.pth"
    elif C.backbone == 'mit_b1':
        C.pretrained_model = "pretrained/segformer/mit_b1.pth"

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'
