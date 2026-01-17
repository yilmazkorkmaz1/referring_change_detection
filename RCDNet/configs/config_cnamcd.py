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

C.seed = 3407

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'CNAM-CD'
C.root_folder = osp.abspath(osp.join(C.root_dir, 'data', 'CNAM-CD'))
C.A_format = '.tif'
C.B_format = '.tif'
C.gt_format = '.tif'
C.is_test = False
C.num_train_imgs = 2000
C.num_eval_imgs = 500
C.num_classes = 6
C.class_names =  ['background', 'impervious surface', 'bare ground', 'vegetation', 'water bodies', 'other']

C.freeze_backbone = False
C.reduce_resolution = False

# Expected dataset layout:
#   root_folder/
#     A/ B/ gt/
#     train.txt val.txt test.txt
C.train_split = 'train'
C.val_split = 'val'
C.test_split = 'test'
C.eval_class_selection = 'first'  # 'first' (deterministic) or 'random'
"""Image Config"""
C.background = 512
C.image_height = 512
C.image_width = 512
# Dataset-specific normalization statistics (averaged A and B)
# A: [0.379, 0.377, 0.343], B: [0.409, 0.402, 0.372]
C.norm_mean = np.array([0.394, 0.390, 0.358])
C.norm_std = np.array([0.212, 0.180, 0.175])
C.use_cached_norm = False
C.use_single_normalization = True

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'sigma_small' # sigma_tiny / sigma_small / sigma_base
C.pretrained_model = None # do not need to change
C.decoder = 'MambaDecoder' # 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 1
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 16
C.train_scale_array = [1]
C.train_scale_array = None
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.use_color_jitter = False
C.jitter_hyper = 0.1

"""Eval Config"""
# C.eval_iter = 1
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] 
C.eval_flip = False
C.eval_crop_size = [512, 512]

"""Store Config"""
C.checkpoint_start_epoch = 5
C.checkpoint_step = 5

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.use_imagenet_pretrain = True
C.trial_name = f"{C.dataset_name}_{C.backbone}_{C.decoder}"
C.log_dir = osp.abspath(osp.join(C.root_dir, "runs", C.trial_name))
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'