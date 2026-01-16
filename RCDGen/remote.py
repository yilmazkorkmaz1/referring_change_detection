import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import PIL.Image as Image
import random
import cv2


def random_mirror(A, B, gt):
    if random.random() >= 0.5:
        A = cv2.flip(A, 1)
        B = cv2.flip(B, 1)
        gt = cv2.flip(gt, 1)

    return A, B, gt


def random_rotation(A, B, gt):
    """Random rotation - geometric augmentation applied to A, B, and GT"""
    if random.random() >= 0.5:
        # Random rotation by 90, 180, or 270 degrees
        k = random.choice([1, 2, 3])
        A = np.rot90(A, k)
        B = np.rot90(B, k)
        gt = np.rot90(gt, k)
    return A, B, gt


def random_color_jitter(A, B):
    """Color jitter - applied only to A and B, not GT"""
    if random.random() >= 0.5:
        # Random brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        A = np.clip(A * brightness_factor, 0, 1)
        B = np.clip(B * brightness_factor, 0, 1)
        
    if random.random() >= 0.5:
        # Random contrast adjustment
        contrast_factor = random.uniform(0.8, 1.2)
        mean_A = np.mean(A, axis=(0, 1), keepdims=True)
        mean_B = np.mean(B, axis=(0, 1), keepdims=True)
        A = np.clip((A - mean_A) * contrast_factor + mean_A, 0, 1)
        B = np.clip((B - mean_B) * contrast_factor + mean_B, 0, 1)
        
    if random.random() >= 0.5:
        # Random saturation adjustment
        saturation_factor = random.uniform(0.8, 1.2)
        # Convert to HSV for saturation adjustment
        A_hsv = cv2.cvtColor((A * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        B_hsv = cv2.cvtColor((B * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        A_hsv[:, :, 1] = np.clip(A_hsv[:, :, 1] * saturation_factor, 0, 255)
        B_hsv[:, :, 1] = np.clip(B_hsv[:, :, 1] * saturation_factor, 0, 255)
        A = cv2.cvtColor(A_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        B = cv2.cvtColor(B_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
    return A, B


def random_vertical_flip(A, B, gt):
    """Vertical flip - geometric augmentation applied to A, B, and GT"""
    if random.random() >= 0.5:
        A = cv2.flip(A, 0)
        B = cv2.flip(B, 0)
        gt = cv2.flip(gt, 0)
    return A, B, gt


class ChangeDataset(Dataset):
    def __init__(self, split_name,A_format,B_format,gt_format,root,class_names, tokenizer, resize=False, upsample=False,**kwargs):
        super(ChangeDataset, self).__init__()
        self._split_name = split_name
        self._A_format = A_format
        self._B_format = B_format
        self._gt_format = gt_format
        self._root_path = root
        self.class_names = class_names
        self._file_names = self._get_file_names(split_name)
        self.tokenizer = tokenizer

        self.resize_transform = transforms.RandomCrop(size=(512, 512))
        self.upsample = upsample
        self.upsample_transform = transforms.Resize((512,512))
        self.resize = resize

    def __len__(self):
        return len(self._file_names)

    def __getitem__(self, index):
        item_name = self._file_names[index]
        A_path = os.path.join(self._root_path, 'A', item_name + self._A_format)
        B_path = os.path.join(self._root_path, 'B', item_name + self._B_format)
        gt_path = os.path.join(self._root_path, 'gt', item_name + self._gt_format)

        A = self._open_image(A_path, "RGB") / 255.0
        B = self._open_image(B_path, "RGB") / 255.0
        gt = self._open_image(gt_path, "L", dtype=np.uint8) 

        if self.resize or self.upsample:
            gt = gt / 255

        
        if self._split_name == 'train':
            if len(np.unique(gt)) == 1:
                class_index = int(np.unique(gt)[0])
            else:
                class_index = np.int16(np.random.choice(np.unique(gt)[1:], 1))[0]
        else:
            class_index = 1
        
        if class_index == 0:
            gt = np.uint8(np.zeros_like(gt))
        else:
            gt = np.uint8(gt == class_index)
    
        caption = "change in " + self.class_names[class_index]

        tokenized = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        A, B, gt = random_mirror(A, B, gt)
        # Additional geometric augmentations (applied to A, B, and GT)
        A, B, gt = random_rotation(A, B, gt)
        A, B, gt = random_vertical_flip(A, B, gt)
        # Color augmentations (applied only to A and B, not GT)
        A, B = random_color_jitter(A, B)
        
        A = torch.from_numpy(np.ascontiguousarray(A.transpose(2, 0, 1))).to(torch.float16)
        B = torch.from_numpy(np.ascontiguousarray(B.transpose(2, 0, 1))).to(torch.float16)
        gt = torch.from_numpy(np.ascontiguousarray(gt)).to(torch.float16)


        if self.resize:
            concatenated = torch.concat([A,B,gt[None]], dim=0)[None]
            concatenated = self.resize_transform(concatenated)[0]
            A = concatenated[0:3]
            B = concatenated[3:6]
            gt = concatenated[6]
        elif self.upsample:
            A = self.upsample_transform(A[None])[0]
            B = self.upsample_transform(B[None])[0]
            gt = self.upsample_transform(gt[None][None])[0][0]

        cond_im = torch.concat([A], dim=0)
        cond_im = (cond_im - 0.5) * 2.0

        image = (torch.concat([B, gt[None]], dim=0))
        image = (image - 0.5) * 2.0

        output_dict = {
            'edited_pixel_values': image, 
            'original_pixel_values': cond_im,
            "input_ids": tokenized.input_ids,
        }

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']
        source = os.path.join(self._root_path, split_name + '.txt')

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            if len(file_name) > 4 and file_name[-4] == '.':
                file_name = file_name[:-4]
            file_names.append(file_name)

        return file_names

    @staticmethod
    def _open_image(filepath, mode="RGB", dtype=None):
        img = np.array(Image.open(filepath).convert(mode), dtype=dtype)
        return img
