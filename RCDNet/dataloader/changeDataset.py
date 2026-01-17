import os
import random
import json
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


def _random_mirror(A: np.ndarray, B: np.ndarray, gt: np.ndarray, rng: random.Random):
    # Horizontal flip
    if rng.random() >= 0.5:
        A = np.flip(A, axis=1).copy()
        B = np.flip(B, axis=1).copy()
        gt = np.flip(gt, axis=1).copy()
    # Vertical flip
    if rng.random() >= 0.5:
        A = np.flip(A, axis=0).copy()
        B = np.flip(B, axis=0).copy()
        gt = np.flip(gt, axis=0).copy()
    return A, B, gt


class ChangeDataset(data.Dataset):
    """
    Open-source dataset for referring change detection.

    - Directory structure:
        root/
          A/ B/ gt/
          train.txt val.txt test.txt   (ids without extension)

    - Output dict keys:
        A: FloatTensor [3,H,W]
        B: FloatTensor [3,H,W]
        gt: FloatTensor [1,H,W]  (binary mask for the selected class)
        text_captions: str | list[str]
            - train: str (selected class caption)
            - val/test: list[str] (captions for all present classes)
        class_indexes: list[int]
            - train: [selected_class_id]
            - val/test: all present class ids (non-background)
        fn: str                  (sample id)
    """

    def __init__(
        self,
        split_name: str,
        A_format: str,
        B_format: str,
        gt_format: str,
        root: str,
        class_names: List[str],
        image_size: Tuple[int, int] = (512, 512),
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        use_color_jitter: bool = False,
        jitter_hyper: float = 0.1,
        eval_class_selection: str = "first",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.split_name = split_name
        self.A_format = A_format
        self.B_format = B_format
        self.gt_format = gt_format
        self.root = root
        self.class_names = class_names
        self.image_size = image_size
        self.eval_class_selection = eval_class_selection

        self._rng = random.Random(seed)

        # Class mapping: maps original GT class indices to unified taxonomy indices
        # e.g., class_mapping = {0: 0, 1: 3, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9}
        # If None, no mapping is applied (original indices used as-is)
        self.class_mapping = kwargs.get("class_mapping", None)

        # Normalize binary GT masks: convert 255 -> 1 for binary CD datasets (LEVIR, WHU, etc.)
        self.normalize_binary_gt = bool(kwargs.get("normalize_binary_gt", False))

        # Legacy synthetic mode support:
        # Some synthetic datasets store separate lists per split:
        #   train_A.txt / train_B.txt / train_gt.txt (and optionally train_synthetic_A.txt)
        # in the same directory as A/ B/ gt/.
        self.synthetic_A = bool(kwargs.get("synthetic_A", False))

        ids_path = os.path.join(self.root, f"{split_name}.txt")
        A_list_path = os.path.join(self.root, f"{split_name}_A.txt")
        B_list_path = os.path.join(self.root, f"{split_name}_B.txt")
        gt_list_path = os.path.join(self.root, f"{split_name}_gt.txt")
        synA_list_path = os.path.join(self.root, f"{split_name}_synthetic_A.txt")

        def _read_list(path: str):
            with open(path, "r") as f:
                return [line.strip().split(".")[0] for line in f if line.strip()]

        if os.path.exists(ids_path):
            self.mode = "single_list"
            self.ids = _read_list(ids_path)
        elif os.path.exists(B_list_path) and os.path.exists(gt_list_path) and (os.path.exists(A_list_path) or os.path.exists(synA_list_path)):
            self.mode = "paired_lists"
            self.A_ids = _read_list(synA_list_path if (self.synthetic_A and os.path.exists(synA_list_path)) else A_list_path)
            self.B_ids = _read_list(B_list_path)
            self.gt_ids = _read_list(gt_list_path)
            if not (len(self.A_ids) == len(self.B_ids) == len(self.gt_ids)):
                raise ValueError(
                    f"List length mismatch for split={split_name}: "
                    f"A={len(self.A_ids)} B={len(self.B_ids)} gt={len(self.gt_ids)} in root={self.root}"
                )
        else:
            raise FileNotFoundError(
                "Could not find split lists. Expected either:\n"
                f"- {ids_path}\n"
                "or (legacy synthetic):\n"
                f"- {A_list_path} (or {synA_list_path})\n"
                f"- {B_list_path}\n"
                f"- {gt_list_path}\n"
            )

        # --- Normalization -------------------------------------------------
        # The older internal dataset (`changedataset_old_clip.py`) computed and cached
        # separate normalization stats for A and B:
        #   <root>/<split>_norm_values.json with keys A_mean/A_std/B_mean/B_std (lists)
        #
        # To keep checkpoints comparable, we optionally load that cache and apply either
        # separate normalization, or a "single normalization" averaged across A/B.
        self.use_cached_norm = bool(kwargs.get("use_cached_norm", False))
        self.use_single_normalization = bool(kwargs.get("use_single_normalization", False))
        norm_cache_file = kwargs.get("norm_cache_file", None)

        self.norm_A = None
        self.norm_B = None
        if self.use_cached_norm:
            cache_path = norm_cache_file or os.path.join(self.root, f"{split_name}_norm_values.json")
            # Some datasets only provide train normalization; fall back to train_norm_values.json
            # when evaluating on val/test.
            if (not os.path.exists(cache_path)) and (norm_cache_file is None) and (split_name != "train"):
                train_cache_path = os.path.join(self.root, "train_norm_values.json")
                if os.path.exists(train_cache_path):
                    cache_path = train_cache_path

            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    norm_values = json.load(f)
                try:
                    mean_A = norm_values["A_mean"]
                    std_A = norm_values["A_std"]
                    mean_B = norm_values["B_mean"]
                    std_B = norm_values["B_std"]
                except KeyError as e:
                    raise KeyError(
                        f"Normalization cache {cache_path!r} is missing key {e!s}. "
                        "Expected keys: A_mean, A_std, B_mean, B_std."
                    ) from e

                if self.use_single_normalization:
                    avg_mean = [(a + b) / 2 for a, b in zip(mean_A, mean_B)]
                    avg_std = [(a + b) / 2 for a, b in zip(std_A, std_B)]
                    self.norm_A = transforms.Normalize(avg_mean, avg_std)
                    self.norm_B = transforms.Normalize(avg_mean, avg_std)
                else:
                    self.norm_A = transforms.Normalize(mean_A, std_A)
                    self.norm_B = transforms.Normalize(mean_B, std_B)
            else:
                raise FileNotFoundError(
                    f"use_cached_norm=True but cache file not found: {cache_path}. "
                    "Expected <split>_norm_values.json (or train_norm_values.json as a fallback)."
                )

        if self.norm_A is None or self.norm_B is None:
            # Default behavior: use provided stats if any, otherwise ImageNet.
            if norm_mean is None or norm_std is None:
                norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            self.norm_A = transforms.Normalize(norm_mean.tolist(), norm_std.tolist())
            self.norm_B = transforms.Normalize(norm_mean.tolist(), norm_std.tolist())

        self.color_jitter = (
            transforms.ColorJitter(
                brightness=jitter_hyper,
                contrast=jitter_hyper,
                saturation=jitter_hyper,
                hue=jitter_hyper,
            )
            if (use_color_jitter and split_name == "train")
            else None
        )

    def __len__(self):
        if getattr(self, "mode", "single_list") == "paired_lists":
            return len(self.B_ids)
        return len(self.ids)

    @staticmethod
    def _open_image(filepath: str, mode: str = "RGB", dtype=None):
        return np.array(Image.open(filepath).convert(mode), dtype=dtype)

    def __getitem__(self, index: int):
        if getattr(self, "mode", "single_list") == "paired_lists":
            A_id = self.A_ids[index]
            B_id = self.B_ids[index]
            gt_id = self.gt_ids[index]
            item_id = B_id
        else:
            item_id = self.ids[index]
            A_id = item_id
            B_id = item_id
            gt_id = item_id

        A_path = os.path.join(self.root, "A", A_id + self.A_format)
        B_path = os.path.join(self.root, "B", B_id + self.B_format)
        gt_path = os.path.join(self.root, "gt", gt_id + self.gt_format)

        A = self._open_image(A_path, "RGB").astype(np.float32) / 255.0
        B = self._open_image(B_path, "RGB").astype(np.float32) / 255.0
        gt = self._open_image(gt_path, "L", dtype=np.uint8)

        # Normalize binary masks: convert 255 -> 1 for binary CD datasets
        # Enabled via config.normalize_binary_gt = True (e.g., LEVIR-CD, WHU-CD)
        if self.normalize_binary_gt and gt.max() > 1:
            gt = (gt > 0).astype(np.uint8)

        # Apply class mapping if provided (for unified taxonomy in mixed training)
        if self.class_mapping is not None:
            gt_mapped = np.zeros_like(gt)
            for orig_idx, unified_idx in self.class_mapping.items():
                gt_mapped[gt == orig_idx] = unified_idx
            gt = gt_mapped

        # Resize to a fixed size (keeps repo assumptions consistent)
        H, W = self.image_size
        if (A.shape[0] != H) or (A.shape[1] != W):
            A = np.array(Image.fromarray((A * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)).astype(np.float32) / 255.0
        if (B.shape[0] != H) or (B.shape[1] != W):
            B = np.array(Image.fromarray((B * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)).astype(np.float32) / 255.0
        if (gt.shape[0] != H) or (gt.shape[1] != W):
            gt = np.array(Image.fromarray(gt).resize((W, H), resample=Image.NEAREST)).astype(np.uint8)

        unique_classes = np.unique(gt)
        unique_classes = unique_classes[unique_classes != 0]  # drop background
        if self.split_name == "train":
            if len(unique_classes) == 0:
                class_index = 0
            else:
                class_index = int(self._rng.choice(unique_classes.tolist()))
            gt_binary = (gt == class_index).astype(np.float32)
            text_caption = self.class_names[int(class_index)] if int(class_index) < len(self.class_names) else str(int(class_index))
            class_indexes = [int(class_index)]
        else:
            # For evaluation, return all present classes to allow multi-class metrics (Score/SeK/Semantic_IoU).
            if len(unique_classes) == 0:
                # background-only; treat as "non-change" (class 0)
                class_indexes = [0]
            else:
                class_indexes = [int(x) for x in unique_classes.tolist()]
            # multi-class gt as int map (not binary)
            gt_binary = gt.astype(np.int64)
            text_caption = [
                self.class_names[int(ci)] if int(ci) < len(self.class_names) else str(int(ci))
                for ci in class_indexes
            ]

        if self.split_name == "train":
            A, B, gt_binary = _random_mirror(A, B, gt_binary, self._rng)

        A_t = torch.from_numpy(np.ascontiguousarray(A.transpose(2, 0, 1))).float()
        B_t = torch.from_numpy(np.ascontiguousarray(B.transpose(2, 0, 1))).float()
        if self.split_name == "train":
            gt_t = torch.from_numpy(np.ascontiguousarray(gt_binary)).float().unsqueeze(0)
        else:
            gt_t = torch.from_numpy(np.ascontiguousarray(gt_binary)).long().unsqueeze(0)

        if self.color_jitter is not None:
            A_t = self.color_jitter(A_t)
            B_t = self.color_jitter(B_t)

        A_t = self.norm_A(A_t)
        B_t = self.norm_B(B_t)

        return {
            "A": A_t,
            "B": B_t,
            "gt": gt_t,
            "text_captions": text_caption,
            "fn": str(item_id),
            "n": len(self),
            "class_indexes": class_indexes,
        }















