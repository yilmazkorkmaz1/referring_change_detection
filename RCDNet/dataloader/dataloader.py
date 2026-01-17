"""
Supported:
- Single dataset root (`config.root_folder = "/path/to/ds"`)
- Mixed training via concatenation (`config.root_folder = ["/path/to/ds1", "/path/to/ds2", ...]`)

For mixed training, you can either:
- provide a list of strings (shared settings across datasets), or
- provide a list of dict specs (per-dataset settings, including different `class_names`).
"""

import os
from torch.utils import data
from torch.utils.data import ConcatDataset

from dataloader.changeDataset import ChangeDataset


def get_dataset(config, split: str):
    """
    Expected dataset layout:
      <root_folder>/
        A/ B/ gt/
        train.txt val.txt test.txt   (each line: sample id, without extension)

    Supports per-split roots via config.root_by_split dict:
      config.root_by_split = {"train": "/path/to/train_data", "val": "/path/to/val_data", ...}
    """
    # Check for per-split root override
    root_by_split = getattr(config, "root_by_split", None)
    if isinstance(root_by_split, dict) and split in root_by_split:
        root = root_by_split[split]
    else:
        root = getattr(config, "root_folder", None)

    if not root:
        raise ValueError("config.root_folder must be set to the dataset root directory.")

    default_A_format = getattr(config, "A_format", ".png")
    default_B_format = getattr(config, "B_format", ".png")
    default_gt_format = getattr(config, "gt_format", ".png")

    # Support per-split class names via config.class_names_by_split
    class_names_by_split = getattr(config, "class_names_by_split", None)
    if isinstance(class_names_by_split, dict) and split in class_names_by_split:
        default_class_names = class_names_by_split[split]
    else:
        default_class_names = getattr(config, "class_names", None)

    if not default_class_names:
        raise ValueError("config.class_names must be set (or provide per-dataset class_names in mixed mode).")

    def _resolve_root_for_split(one_root: str) -> str:
        """
        Support two common dataset layouts:

        Layout A (flat):
          root/
            A/ B/ gt/
            train.txt val.txt test.txt

        Layout B (split subfolders):
          root/
            train/ (contains A/ B/ gt/ + train.txt)
            val/   (contains A/ B/ gt/ + val.txt)
            test/  (contains A/ B/ gt/ + test.txt)
        """
        def _looks_like_dataset_root(path: str) -> bool:
            return (
                os.path.isdir(os.path.join(path, "A"))
                and os.path.isdir(os.path.join(path, "B"))
                and os.path.isdir(os.path.join(path, "gt"))
            )

        # If flat layout exists *and* the image folders exist, use it
        if _looks_like_dataset_root(one_root):
            if os.path.exists(os.path.join(one_root, f"{split}.txt")):
                return one_root
            if os.path.exists(os.path.join(one_root, f"{split}_B.txt")) and os.path.exists(os.path.join(one_root, f"{split}_gt.txt")):
                return one_root

        # If split subfolder layout exists, use <root>/<split>
        split_root = os.path.join(one_root, split)
        if os.path.isdir(split_root):
            # Prefer split_root if it looks like a dataset root (A/B/gt folders) or has the split lists/norm cache.
            if _looks_like_dataset_root(split_root):
                return split_root
            if os.path.exists(os.path.join(split_root, f"{split}_norm_values.json")):
                return split_root
            if os.path.exists(os.path.join(split_root, f"{split}.txt")):
                return split_root
            if os.path.exists(os.path.join(split_root, f"{split}_B.txt")) and os.path.exists(os.path.join(split_root, f"{split}_gt.txt")):
                return split_root

        # Some datasets store both val/test under a shared folder, e.g.:
        #   root/evaluation/{A,B,gt,val.txt,test.txt,...}
        if split in ("val", "test"):
            eval_root = os.path.join(one_root, "evaluation")
            if os.path.isdir(eval_root) and _looks_like_dataset_root(eval_root):
                if os.path.exists(os.path.join(eval_root, f"{split}.txt")):
                    return eval_root
                if os.path.exists(os.path.join(eval_root, f"{split}_B.txt")) and os.path.exists(os.path.join(eval_root, f"{split}_gt.txt")):
                    return eval_root
                if os.path.exists(os.path.join(eval_root, f"{split}_norm_values.json")):
                    return eval_root

        # Some datasets keep *all* splits under a single subfolder (commonly "train"),
        # e.g. root/train/{A,B,gt,train.txt,val.txt,...}
        for candidate in ("train", "val", "test", "evaluation"):
            candidate_root = os.path.join(one_root, candidate)
            if not os.path.isdir(candidate_root):
                continue
            if not _looks_like_dataset_root(candidate_root):
                continue
            if os.path.exists(os.path.join(candidate_root, f"{split}.txt")):
                return candidate_root
            if os.path.exists(os.path.join(candidate_root, f"{split}_B.txt")) and os.path.exists(os.path.join(candidate_root, f"{split}_gt.txt")):
                return candidate_root

        return one_root

    def _make_one(
        one_root: str,
        *,
        A_format: str = default_A_format,
        B_format: str = default_B_format,
        gt_format: str = default_gt_format,
        class_names=default_class_names,
        norm_mean=None,
        norm_std=None,
        image_size=None,
        synthetic_A: bool = False,
        class_mapping=None,
    ):
        one_root = _resolve_root_for_split(one_root)
        return ChangeDataset(
            split_name=split,
            A_format=A_format,
            B_format=B_format,
            gt_format=gt_format,
            root=one_root,
            class_names=class_names,
            image_size=image_size or (config.image_height, config.image_width),
            norm_mean=norm_mean if norm_mean is not None else getattr(config, "norm_mean", None),
            norm_std=norm_std if norm_std is not None else getattr(config, "norm_std", None),
            use_color_jitter=getattr(config, "use_color_jitter", False),
            jitter_hyper=getattr(config, "jitter_hyper", 0.1),
            eval_class_selection=getattr(config, "eval_class_selection", "first"),
            seed=getattr(config, "seed", 0),
            # Legacy normalization compatibility (for checkpoints trained with changedataset_old_clip.py)
            use_cached_norm=getattr(config, "use_cached_norm", False),
            use_single_normalization=getattr(config, "use_single_normalization", False),
            norm_cache_file=getattr(config, "norm_cache_file", None),
            synthetic_A=synthetic_A,
            class_mapping=class_mapping,
            # Binary GT normalization (255 -> 1) for LEVIR, WHU, etc.
            normalize_binary_gt=getattr(config, "normalize_binary_gt", False),
        )

    if isinstance(root, (list, tuple)):
        if len(root) == 0:
            raise ValueError("config.root_folder list is empty.")
        # Mixed mode with per-dataset settings:
        # config.root_folder = [
        #   {"root": "...", "class_names": [...], "A_format": ".png", "B_format": ".png", "gt_format": ".png"},
        #   {"root": "...", "class_names": [...], "A_format": ".tif", "B_format": ".tif", "gt_format": ".tif"},
        # ]
        if isinstance(root[0], dict):
            datasets = []
            for spec in root:
                # Allow different roots per split (common in legacy synthetic datasets)
                root_by_split = spec.get("root_by_split", None)
                if isinstance(root_by_split, dict):
                    one_root = root_by_split.get(split) or root_by_split.get(f"{split}_root")
                else:
                    one_root = spec.get("root")
                if not one_root:
                    raise ValueError(f"Mixed dataset spec must provide `root` or `root_by_split` for split={split!r}.")
                # Support per-split format overrides (e.g., synthetic uses .png, real uses .tif)
                A_fmt_by_split = spec.get("A_format_by_split", {})
                B_fmt_by_split = spec.get("B_format_by_split", {})
                gt_fmt_by_split = spec.get("gt_format_by_split", {})
                A_format = A_fmt_by_split.get(split, spec.get("A_format", default_A_format))
                B_format = B_fmt_by_split.get(split, spec.get("B_format", default_B_format))
                gt_format = gt_fmt_by_split.get(split, spec.get("gt_format", default_gt_format))

                datasets.append(
                    _make_one(
                        one_root,
                        A_format=A_format,
                        B_format=B_format,
                        gt_format=gt_format,
                        class_names=spec.get("class_names", default_class_names),
                        norm_mean=spec.get("norm_mean", None),
                        norm_std=spec.get("norm_std", None),
                        image_size=spec.get("image_size", None),
                        # pass through any dataset-mode flags (e.g., synthetic_A)
                        synthetic_A=spec.get("synthetic_A", False),
                        # class mapping for unified taxonomy in mixed training
                        class_mapping=spec.get("class_mapping", None),
                    )
                )
            concat_ds = ConcatDataset(datasets)
            # Preserve class_names for evaluation (use unified taxonomy from config if available)
            concat_ds.class_names = getattr(config, "unified_class_names", None) or default_class_names
            return concat_ds

        # Mixed mode with shared settings:
        datasets = [_make_one(r) for r in root]
        concat_ds = ConcatDataset(datasets)
        concat_ds.class_names = getattr(config, "unified_class_names", None) or default_class_names
        return concat_ds

    return _make_one(root)


def _get_train_loader(config):
    dataset = get_dataset(config, split=getattr(config, "train_split", "train"))
    return data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=getattr(config, "num_workers", 4),
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )


def get_val_loader(config):
    return get_dataset(config, split=getattr(config, "val_split", "val"))


def get_train_loader(*args, **kwargs):
    """
    Legacy signature supported:
      get_train_loader(accelerator, dataset_cls, config)
    New signature:
      get_train_loader(config)
    """
    if len(args) == 1 and not kwargs:
        return _get_train_loader(args[0])
    if len(args) >= 3:
        config = args[2]
        return _get_train_loader(config)
    if "config" in kwargs:
        return _get_train_loader(kwargs["config"])
    raise TypeError("Unsupported get_train_loader call. Expected (config) or (accelerator, dataset_cls, config).")

