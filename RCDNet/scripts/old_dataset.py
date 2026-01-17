import os
from pickletools import uint8
import cv2
import torch
import numpy as np
import clip
import PIL.Image as Image
import torch.utils.data as data
from torchvision import transforms
import random
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import islice
import logging
from transformers import CLIPTokenizer

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def random_mirror(A, B, gt):
    if random.random() >= 0.5:
        A = cv2.flip(A, 1)
        B = cv2.flip(B, 1)
        gt = cv2.flip(gt, 1)
    if random.random() >= 0.5:
        A = cv2.flip(A, 0)
        B = cv2.flip(B, 0)
        gt = cv2.flip(gt, 0)

    return A, B, gt

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def random_rotation(A, B, gt, max_angle=10):
    """
    Randomly rotate the images and ground truth by a small angle.
    
    Args:
    A, B, gt: Input images and ground truth
    max_angle: Maximum rotation angle in degrees
    
    Returns:
    Rotated A, B, and gt
    """
    angle = random.uniform(-max_angle, max_angle)
    
    height, width = A.shape[:2]
    center = (width / 2, height / 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    A_rotated = cv2.warpAffine(A, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    B_rotated = cv2.warpAffine(B, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gt_rotated = cv2.warpAffine(gt, rotation_matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    
    return A_rotated, B_rotated, gt_rotated

def random_mirror_and_rotate(A, B, gt):
    # First apply random mirroring
    A, B, gt = random_mirror(A, B, gt)
        
    return A, B, gt

class ChangeDataset(data.Dataset):
    def __init__(self, split_name, A_format, B_format, gt_format, root, class_names, 
                 resize=False, upsample=False, use_class_frequencies=False, 
                 synthetic_dataset=False, synthetic_A=False, broken_B=False,
                 blur_synthetic=False, min_kernel_size=3, max_kernel_size=9, 
                 min_sigma=0.5, max_sigma=4.0, use_single_normalization=False,
                 use_color_jitter=False, num_samples=None, selected_indices_file=None,
                 synthetic_B_folder=None, vae_A=False, mask_B=False, jitter_hyper=0.1, missing_classes_fill=False):  # Added mask_B parameter
        super(ChangeDataset, self).__init__()
        self._split_name = split_name
        self._A_format = A_format
        self._B_format = B_format
        self._gt_format = gt_format
        self._root_path = root
        self.class_names = class_names
        self.synthetic_dataset = synthetic_dataset
        self.synthetic_A = synthetic_A
        self.resize = resize
        self.upsample = upsample
        self.use_class_frequencies = use_class_frequencies
        self.broken_B = broken_B  
        self.blur_synthetic = blur_synthetic
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.use_single_normalization = use_single_normalization
        self.use_color_jitter = use_color_jitter
        self.selected_indices_file = selected_indices_file
        self.synthetic_B_folder = synthetic_B_folder
        self.vae_A = vae_A  # New attribute for vae_A
        self.mask_B = mask_B  # New attribute for masking B images
        self.num_samples = num_samples
        
        

        if synthetic_dataset:
            if missing_classes_fill:
                self._A_files = self._get_file_names(f'{split_name}_A_high_iou.txt')
                self._B_files = self._get_file_names(f'{split_name}_B_high_iou.txt')
                self._gt_files = self._get_file_names(f'{split_name}_gt_high_iou.txt')
            else:
                if synthetic_A:
                    self._A_files = self._get_file_names(f'{split_name}_synthetic_A.txt')
                else:
                    self._A_files = self._get_file_names(f'{split_name}_A.txt')
                self._B_files = self._get_file_names(f'{split_name}_B.txt')
                self._gt_files = self._get_file_names(f'{split_name}_gt.txt')
        else:
            self._file_names = self._get_file_names(f'{split_name}.txt')
        
        if self.selected_indices_file is not None:
            self._apply_selected_indices()
        
        # Apply num_samples limit if specified
        if self.num_samples is not None:
            if synthetic_dataset:
                combined = list(zip(self._A_files, self._B_files, self._gt_files))
                sampled = random.sample(combined, min(self.num_samples, len(combined)))
                self._A_files, self._B_files, self._gt_files = zip(*sampled)
            else:
                self._file_names = random.sample(self._file_names, min(self.num_samples, len(self._file_names)))
        
        # Replace clip.tokenize with CLIPTokenizer
     #   self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        if split_name == "train":
            self.resize_transform = transforms.RandomResizedCrop(size=(512, 512), scale=(0.5, 0.5))
        else:
            self.resize_transform = transforms.Resize((512, 512))
        self.upsample = upsample
        self.upsample_transform = transforms.Resize((512, 512))
        self.resize = resize
        self.use_class_frequencies = use_class_frequencies
        
        if self.use_class_frequencies:
            self.class_frequencies = self._calculate_class_frequencies()
        
        # Load or calculate normalization values
        self.norm_values = self._load_or_calculate_norm_values()
        # Ensure synthetic_A_mean and synthetic_A_std are present if needed
        if synthetic_dataset and synthetic_A:
            if 'synthetic_A_mean' not in self.norm_values or 'synthetic_A_std' not in self.norm_values:
                print("Warning: synthetic_A normalization values not found. Recalculating...")
                self.norm_values = self._calculate_norm_values()  # Recalculate if missing
            self.norm_A = transforms.Normalize(self.norm_values['synthetic_A_mean'], self.norm_values['synthetic_A_std'])
        else:
            if 'A_mean' not in self.norm_values or 'A_std' not in self.norm_values:
                print("Warning: A normalization values not found. Recalculating...")
                self.norm_values = self._calculate_norm_values()  # Recalculate if missing
            self.norm_A = transforms.Normalize(self.norm_values['A_mean'], self.norm_values['A_std'])
        self.norm_B = transforms.Normalize(self.norm_values['B_mean'], self.norm_values['B_std'])

        # New condition for mask_B
        if self.mask_B:
            if synthetic_dataset and synthetic_A:
                self.norm_B = transforms.Normalize(self.norm_values['synthetic_A_mean'], self.norm_values['synthetic_A_std'])
            else:
                self.norm_B = transforms.Normalize(self.norm_values['A_mean'], self.norm_values['A_std'])
            print("Using A normalization for B due to masking")

        # New code for single normalization option
        if self.use_single_normalization:
            if synthetic_dataset and synthetic_A:
                mean_A = self.norm_values['synthetic_A_mean']
                std_A = self.norm_values['synthetic_A_std']
            else:
                mean_A = self.norm_values['A_mean']
                std_A = self.norm_values['A_std']
            mean_B = self.norm_values['B_mean']
            std_B = self.norm_values['B_std']
            
            # Calculate average mean and std
            avg_mean = [(a + b) / 2 for a, b in zip(mean_A, mean_B)]
            avg_std = [(a + b) / 2 for a, b in zip(std_A, std_B)]
            
            self.norm_A = transforms.Normalize(avg_mean, avg_std)
            self.norm_B = transforms.Normalize(avg_mean, avg_std)
            print("Using single normalization for both A and B images")
        else:
            print("Using separate normalization for A and B images")

        if self.use_color_jitter and self._split_name == 'train':
            self.color_jitter = transforms.ColorJitter(brightness=jitter_hyper, contrast=jitter_hyper, saturation=jitter_hyper, hue=jitter_hyper)
        else:
            self.color_jitter = None




    def _load_or_calculate_class_frequencies(self):
        cache_file = os.path.join(self._root_path, f'{self._split_name}_class_frequencies.json')
        
        if os.path.exists(cache_file):
            print(f"Loading cached class frequencies from {cache_file}")
            with open(cache_file, 'r') as f:
                return {int(k): v for k, v in json.load(f).items()}
        else:
            print("Calculating class frequencies...")
            frequencies = self._calculate_class_frequencies()
            
            print(f"Caching class frequencies to {cache_file}")
            with open(cache_file, 'w') as f:
                json.dump({str(k): v for k, v in frequencies.items()}, f)
            
            return frequencies
    
    def _calculate_class_frequencies(self):
        # Calculate class counts at image level
        class_counts = {cls: 0 for cls in range(len(self.class_names))}
        total_images = 0

        if self.synthetic_dataset:
            sampled_files = self._gt_files
        else:
            sampled_files = self._file_names

        for item_name in tqdm(sampled_files, desc="Calculating class frequencies"):
            gt_path = os.path.join(self._root_path, 'gt', item_name + self._gt_format)
            gt = self._open_image(gt_path, "L", dtype=np.uint8)
            
            # Count classes present in the image
            unique_classes = np.unique(gt)
            for cls in unique_classes:
                if cls != 0:  # Exclude background class if needed
                    class_counts[cls] += 1
            total_images += 1

        # Print class counts
        print("Class Counts (number of images containing each class):")
        for cls, count in class_counts.items():
            print(f"Class {cls}: {count}")
        
        # Print total number of images
        print(f"Total images: {total_images}")

        # Calculate frequencies (optional, in case you still need them)
        class_frequencies = {cls: count / total_images for cls, count in class_counts.items()}
        
        return class_frequencies

    def _load_or_calculate_norm_values(self):
        if self.selected_indices_file:
            cache_file = os.path.join(self._root_path, 'selected_subset_norm_values_entropy.json')
        else:
            cache_file = os.path.join(self._root_path, f'{self._split_name}_norm_values.json')
        
        if os.path.exists(cache_file):
            print(f"Loading cached normalization values from {cache_file}")
            with open(cache_file, 'r') as f:
                norm_values = json.load(f)
            
            # Check if all required values are present
            required_keys = ['A_mean', 'A_std', 'B_mean', 'B_std']
            if self.synthetic_dataset and self.synthetic_A:
                required_keys.extend(['synthetic_A_mean', 'synthetic_A_std'])
            
            if all(key in norm_values for key in required_keys):
                return norm_values
            else:
                print("Cached normalization values are incomplete. Recalculating...")
        else:
            print("Normalization values not found. Calculating...")
        
        norm_values = self._calculate_norm_values()
        
        print(f"Caching normalization values to {cache_file}")
        with open(cache_file, 'w') as f:
            json.dump(norm_values, f)
        
        return norm_values

    def _calculate_norm_values(self):
        A_sum = np.zeros(3)
        B_sum = np.zeros(3)
        A_sum_sq = np.zeros(3)
        B_sum_sq = np.zeros(3)
        synthetic_A_sum = np.zeros(3)
        synthetic_A_sum_sq = np.zeros(3)
        total_pixels = 0
        synthetic_A_total_pixels = 0

        # Determine the number of samples to use (10% of the dataset)
        num_samples = len(self._file_names if not self.synthetic_dataset else self._B_files)
        sample_size = max(int(num_samples * 0.1), 100)  # Ensure at least 100 samples

        # Randomly select 10% of the files
        if self.synthetic_dataset:
            if self.synthetic_A:
                sampled_files = random.sample(list(zip(self._A_files, self._B_files)), sample_size)
            else:
                sampled_files = random.sample(list(zip(self._A_files, self._B_files)), sample_size)
        else:
            sampled_files = random.sample(self._file_names, sample_size)

        problematic_files = []

        for item in tqdm(sampled_files, desc="Calculating normalization values"):
            if self.synthetic_dataset:
                if self.synthetic_A:
                    A_path = os.path.join(self._root_path, 'B', item[0] + self._A_format)
                else:
                    A_path = os.path.join(self._root_path, 'A', item[0] + self._A_format)
                B_path = os.path.join(self._root_path, 'B', item[1] + self._B_format)
            else:
                A_path = os.path.join(self._root_path, 'A', item + self._A_format)
                B_path = os.path.join(self._root_path, 'B', item + self._B_format)

            try:
                A = self._open_image(A_path, "RGB") / 255.0
                B = self._open_and_resize_B(B_path) if self.broken_B else self._open_image(B_path, "RGB") / 255.0

                B_sum += np.sum(B, axis=(0, 1))
                B_sum_sq += np.sum(np.square(B), axis=(0, 1))
                total_pixels += A.shape[0] * A.shape[1]

                if self.synthetic_dataset and self.synthetic_A:
                    synthetic_A_sum += np.sum(A, axis=(0, 1))
                    synthetic_A_sum_sq += np.sum(np.square(A), axis=(0, 1))
                    synthetic_A_total_pixels += A.shape[0] * A.shape[1]
                else:
                    A_sum += np.sum(A, axis=(0, 1))
                    A_sum_sq += np.sum(np.square(A), axis=(0, 1))

                # Check for NaN or inf values after each update
                if np.any(np.isnan(A_sum)) or np.any(np.isinf(A_sum)) or \
                   np.any(np.isnan(B_sum)) or np.any(np.isinf(B_sum)) or \
                   np.any(np.isnan(A_sum_sq)) or np.any(np.isinf(A_sum_sq)) or \
                   np.any(np.isnan(B_sum_sq)) or np.any(np.isinf(B_sum_sq)):
                    logging.warning(f"NaN or inf values detected after processing {A_path} and {B_path}")
                    logging.warning(f"A_sum change: {A_sum - A_sum_before}")
                    logging.warning(f"B_sum change: {B_sum - B_sum_before}")
                    logging.warning(f"A_sum_sq change: {A_sum_sq - A_sum_sq_before}")
                    logging.warning(f"B_sum_sq change: {B_sum_sq - B_sum_sq_before}")
                    problematic_files.append((A_path, B_path))

            except Exception as e:
                logging.error(f"Error processing image {A_path} or {B_path}: {str(e)}")
                problematic_files.append((A_path, B_path))

        A_mean = A_sum / total_pixels
        B_mean = B_sum / total_pixels

        try:
            A_std = np.sqrt(A_sum_sq / total_pixels - np.square(A_mean))
        except Exception as e:
            logging.error(f"Error calculating A_std: {str(e)}")
            logging.error(f"A_sum_sq: {A_sum_sq}, total_pixels: {total_pixels}, A_mean: {A_mean}")
            raise

        try:
            B_std = np.sqrt(B_sum_sq / total_pixels - np.square(B_mean))
        except Exception as e:
            logging.error(f"Error calculating B_std: {str(e)}")
            logging.error(f"B_sum_sq: {B_sum_sq}, total_pixels: {total_pixels}, B_mean: {B_mean}")
            raise

        # Check for NaN or infinite values in final results
        if np.any(np.isnan(A_std)) or np.any(np.isinf(A_std)):
            logging.warning(f"A_std contains NaN or inf values: {A_std}")
        if np.any(np.isnan(B_std)) or np.any(np.isinf(B_std)):
            logging.warning(f"B_std contains NaN or inf values: {B_std}")

        if problematic_files:
            logging.warning("Problematic files detected:")
            for a_path, b_path in problematic_files:
                logging.warning(f"A: {a_path}, B: {b_path}")

        
        result = {
            'B_mean': B_mean.tolist(),
            'B_std': B_std.tolist(),
        }

        if self.synthetic_dataset and self.synthetic_A:
            synthetic_A_mean = synthetic_A_sum / synthetic_A_total_pixels
            synthetic_A_std = np.sqrt(synthetic_A_sum_sq / synthetic_A_total_pixels - np.square(synthetic_A_mean))
            result.update({
                'synthetic_A_mean': synthetic_A_mean.tolist(),
                'synthetic_A_std': synthetic_A_std.tolist()
            })
        else:
            result.update({
                'A_mean': A_mean.tolist(),
                'A_std': A_std.tolist(),
            })

        return result

    def apply_blur(self, image):
        kernel_size = random.randrange(self.min_kernel_size, self.max_kernel_size + 1, 2)
        sigma = random.uniform(self.min_sigma, self.max_sigma)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def __len__(self):
        if self.synthetic_dataset:
            return len(self._B_files)
        else:
            return len(self._file_names)

    def __getitem__(self, index):
        if self.synthetic_dataset:
            item_name = self._B_files[index]
            if self.synthetic_A:
                if self.vae_A:
                    A_path = os.path.join(self._root_path, 'vae_recon_A', self._A_files[index] + self._A_format)
                else:
                    A_path = os.path.join(self._root_path, 'B', self._A_files[index] + self._A_format)
            else:
                A_path = os.path.join(self._root_path, 'A', self._A_files[index] + self._A_format)

            if self.synthetic_B_folder is not None:
                B_path = os.path.join(self.synthetic_B_folder, self._B_files[index] + self._B_format)
            else:
                B_path = os.path.join(self._root_path, 'B', self._B_files[index] + self._B_format)

            gt_path = os.path.join(self._root_path, 'gt', self._gt_files[index] + self._gt_format)
        else:
            item_name = self._file_names[index]
            A_path = os.path.join(self._root_path, 'A', item_name + self._A_format)
            B_path = os.path.join(self._root_path, 'B', item_name + self._B_format)
            gt_path = os.path.join(self._root_path, 'gt', item_name + self._gt_format)

        A = self._open_image(A_path, "RGB") / 255.0
        B = self._open_and_resize_B(B_path) if self.broken_B else self._open_image(B_path, "RGB") / 255.0
        gt = self._open_image(gt_path, "L", dtype=np.uint8)

        if self.synthetic_dataset and self.mask_B:
            # Convert gt to float and normalize to [0, 1]
            mask = (gt > 0).astype(np.float32)
            # Expand mask dimensions to match A and B
            mask = np.expand_dims(mask, axis=-1)
            # Apply masking: B = A * (1-mask) + B * mask
            B = A * (1 - mask) + B * mask

        if self.synthetic_dataset and self.blur_synthetic:
            A = A.astype(np.float32)  # Ensure float type for blurring
            B = B.astype(np.float32)
            A = self.apply_blur(A)
            B = self.apply_blur(B)
            A = np.clip(A, 0, 1)  # Clip values to [0, 1] range
            B = np.clip(B, 0, 1)

        if self.synthetic_dataset:
            # Handle potentially corrupted GT images
            unique_values = np.unique(gt)
            if len(unique_values) > 2:  # More than two classes (including background)
                max_class = np.max(unique_values)
                gt[gt != 0] = max_class  # Set all non-zero values to the maximum class value

        if self.resize or self.upsample or "LEVIR" in self._root_path:
            gt = np.ceil(gt / 255).astype(np.uint8)

        unique_classes = np.unique(gt)
        if not (len(unique_classes) == 1 and unique_classes[0] == 0):
            unique_classes = unique_classes[unique_classes != 0]  # Remove background class

        if self._split_name == 'train':
            if self.use_class_frequencies:
                try:
                    inv_frequencies = [1 / self.class_frequencies[cls] for cls in unique_classes]
                    probs = inv_frequencies / np.sum(inv_frequencies)
                except:
                    print(A_path)
                    print(unique_classes)
                    probs = None  # This will result in equal probabilities in np.random.choice
            else:
                probs = None  # This will result in equal probabilities in np.random.choice
            
            class_index = np.random.choice(unique_classes, p=probs)
            if class_index == 0:
                gt_binary = np.zeros_like(gt).astype(np.uint8)
            else:
                gt_binary = (gt == class_index).astype(np.uint8)  # Binarize for the selected class
            
            # Simulate classifier-free guidance
            if random.random() < 0.0:
             #   caption = self.tokenizer("", padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                text_caption = ""
            else:
             #   caption = self.tokenizer(self.class_names[int(class_index)], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                text_caption = self.class_names[int(class_index)]
        else:
            gt_binary = gt
          #  captions = [self.tokenizer(self.class_names[int(cls)], padding="max_length", max_length=77, truncation=True, return_tensors="pt") for cls in unique_classes]
            text_captions = [self.class_names[int(cls)] for cls in unique_classes]

        if self._split_name == 'train':
            A, B, gt_binary = random_mirror_and_rotate(A, B, gt_binary)

        A = torch.from_numpy(np.ascontiguousarray(A.transpose(2, 0, 1))).to(torch.float32)
        B = torch.from_numpy(np.ascontiguousarray(B.transpose(2, 0, 1))).to(torch.float32)
        gt_tensor = torch.from_numpy(np.ascontiguousarray(gt_binary if self._split_name == 'train' else gt)).to(torch.float32)

        if self.use_color_jitter and self._split_name == 'train':
            A = self.color_jitter(A)
            B = self.color_jitter(B)

        A = self.norm_A(A)
        B = self.norm_B(B)

        if self.resize:
            concatenated = torch.concat([A, B, gt_tensor[None]], dim=0)[None]
            concatenated = self.resize_transform(concatenated)[0]
            A = concatenated[0:3]
            B = concatenated[3:6]
            gt_tensor = concatenated[6:7]
        elif self.upsample:
            A = self.upsample_transform(A[None])[0]
            B = self.upsample_transform(B[None])[0]
            if self._split_name == 'train':
                gt_tensor = self.upsample_transform(gt_tensor[None][None])[0]
            else:
                gt_tensor = gt_tensor[None]
        else:
            gt_tensor = gt_tensor[None]


        if self._split_name == 'train':
            gt_tensor[gt_tensor > 1] = 1  # Ensure valid class indices for binary gt

        output_dict = {
            'A': A,
            'B': B,
            'gt': gt_tensor,
           # 'captions': caption['input_ids'].squeeze(0) if self._split_name == 'train' else [c['input_ids'].squeeze(0) for c in captions],
           # 'attention_mask': caption['attention_mask'].squeeze(0) if self._split_name == 'train' else [c['attention_mask'].squeeze(0) for c in captions],
            'text_captions': text_caption if self._split_name == 'train' else text_captions,
            'fn': str(item_name),
            'n': self.__len__(),
            'class_indexes': [int(class_index)] if self._split_name == 'train' else unique_classes.tolist()
        }

        return output_dict

    def _get_file_names(self, split_name):
        source = os.path.join(self._root_path, split_name)

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            if len(file_name) > 4:
                if file_name[-4] == '.':
                    file_name = file_name[:-4]
            file_names.append(file_name)

        return file_names

    def _apply_selected_indices(self):
        selected_indices = np.load(self.selected_indices_file)
        if self.synthetic_dataset:
            self._A_files = [self._A_files[i] for i in selected_indices]
            self._B_files = [self._B_files[i] for i in selected_indices]
            self._gt_files = [self._gt_files[i] for i in selected_indices]
        else:
            self._file_names = [self._file_names[i] for i in selected_indices]

        # Apply num_samples limit after selecting indices if specified
        if self.num_samples is not None:
            if self.synthetic_dataset:
                self._A_files = self._A_files[:self.num_samples]
                self._B_files = self._B_files[:self.num_samples]
                self._gt_files = self._gt_files[:self.num_samples]
            else:
                self._file_names = self._file_names[:self.num_samples]

    def get_length(self):
       # return 100
       
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode="RGB", dtype=None):
        img = np.array(Image.open(filepath).convert(mode), dtype=dtype)
        return img

    def _open_and_resize_B(self, filepath):
        """Open B image and resize it to match A image size."""
        B = self._open_image(filepath, "RGB") / 255.0
        B = cv2.resize(B, (512, 512), interpolation=cv2.INTER_LINEAR)
        return B

def analyze_dataset(dataset, num_samples=300):
    item_a_array = []
    item_b_array = []
    gt_array = []
    captions = []

    # # Get total number of samples in the dataset
    # total_samples = len(dataset)

    # # Generate random indices
    # random_indices = random.sample(range(total_samples), min(num_samples, total_samples))

    # for idx in tqdm(random_indices, desc="Analyzing dataset", total=len(random_indices)):
    #     item = dataset[idx]
    #     im_a = item['A'].permute(1,2,0).cpu().numpy()
    #     im_b = item['B'].permute(1,2,0).cpu().numpy()
    #     im_gt = item['gt'].permute(1,2,0).cpu().numpy()
    #     item_a_array.append(im_a)
    #     item_b_array.append(im_b)
    #     gt_array.append(im_gt)
    #     caption = item["text_captions"] if isinstance(item["text_captions"], list) else [item["text_captions"]]
    #     captions.extend(caption)
        
    # item_a_array = np.asarray(item_a_array) 
    # item_b_array = np.asarray(item_b_array)
    # gt_array = np.asarray(gt_array)

    # print("Image A statistics:")
    # for i in range(3):
    #     print(f"Channel {i}: Mean = {np.mean(item_a_array[:, :,:,i]):.4f}, Std = {np.std(item_a_array[:,:,:,i]):.4f}")

    # print("\nImage B statistics:")
    # for i in range(3):
    #     print(f"Channel {i}: Mean = {np.mean(item_b_array[:, :,:,i]):.4f}, Std = {np.std(item_b_array[:,:,:,i]):.4f}")

    # unique_classes = np.unique(gt_array)
    # class_counts = {cls: 0 for cls in unique_classes}
    # total_images = len(gt_array)

    # for gt_image in gt_array:
    #     classes_in_image = np.unique(gt_image)
    #     for cls in classes_in_image:
    #         class_counts[cls] += 1

    # print("\nGround Truth statistics:")
    # for cls, count in class_counts.items():
    #     print(f"Class {cls}: present in {count} images")

    # print(f"\nUnique captions ({len(set(captions))}):")
    # for caption in set(captions):
    #     print(f"- {caption}")

    # print("\nNormalization method:")
    # if dataset.use_single_normalization:
    #     print("Using single normalization for both A and B images")
    # else:
    #     print("Using separate normalization for A and B images")

    # print("\nAugmentation:")
    # if dataset.use_color_jitter:
    #     print("Using ColorJitter augmentation")
    # else:
    #     print("No ColorJitter augmentation")

    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(class_counts)), list(class_counts.values()))
    # plt.title("Class Distribution (Number of Images)")
    # plt.xlabel("Class Index")
    # plt.ylabel("Number of Images")
    # plt.xticks(range(len(class_counts)), list(class_counts.keys()))
    # plt.savefig("class_distribution.png")
    # plt.close()

    # Random sampling of 100 indices
    num_samples = 100
    random_indices = random.sample(range(len(dataset)), num_samples)
    
    # Create output directory if it doesn't exist
    output_dir = "sample_pairs"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, idx in enumerate(random_indices):
        sample = dataset[idx]
        
        # Create a new figure for each sample
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Get captions
        captions = sample['text_captions']
        if isinstance(captions, list):
            caption_text = ", ".join(captions)
        else:
            caption_text = captions

        # Denormalize A
        if not dataset.synthetic_A:
            A_denorm = denormalize(sample['A'].clone(), dataset.norm_values['A_mean'], dataset.norm_values['A_std'])
        else:
            A_denorm = denormalize(sample['A'].clone(), dataset.norm_values['synthetic_A_mean'], dataset.norm_values['synthetic_A_std'])

        axes[0].imshow(A_denorm.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        axes[0].set_title(f"A: {caption_text}", fontsize=8)
        
        # Denormalize B
        B_denorm = denormalize(sample['B'].clone(), dataset.norm_values['B_mean'], dataset.norm_values['B_std'])
        axes[1].imshow(B_denorm.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        axes[1].set_title("B", fontsize=8)
        
        # Ground truth doesn't need denormalization
        gt_np = sample['gt'].squeeze().cpu().numpy()
        axes[2].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("Ground Truth", fontsize=8)
        
        # Remove x and y ticks for cleaner look
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{i:02d}.png"), bbox_inches='tight')
        plt.close()

    print(f"Saved {len(random_indices)} sample pairs in the '{output_dir}' directory")

def visualize_gt_samples(dataset, num_samples=50):
    # Randomly select indices
    total_samples = len(dataset)

    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))

    # Create a grid of subplots
    rows = int(num_samples**0.5)
    cols = (num_samples + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()
    
    for i, idx in enumerate(tqdm(sample_indices, desc="Visualizing GT samples")):
        sample = dataset[idx]
        gt = sample['gt'].squeeze().cpu().numpy()
        print(gt.shape)
       # print(np.max(gt))
        axes[i].imshow(gt, cmap='gray')
        axes[i].axis('off')
        
        # Add caption if available
        if 'text_captions' in sample:
            caption = sample['text_captions']
            if isinstance(caption, list):
                caption = ', '.join(caption)
            axes[i].set_title(caption, fontsize=6)

    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("gt_samples.png", dpi=300)
    plt.close()

    print(f"GT samples visualization saved as 'gt_samples.png'")

def visualize_multiclass_gt_samples(dataset, num_samples=10):
    total_samples = len(dataset)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))

    for idx in tqdm(sample_indices, desc="Visualizing GT samples"):
        sample = dataset[idx]
        gt = sample['gt'].squeeze().cpu().numpy()
        
        unique_classes = np.unique(gt)
        print(unique_classes)
        num_classes = len(unique_classes)
        
        # Create a figure with subplots for each class plus the original GT
        fig, axes = plt.subplots(1, num_classes + 1, figsize=(3 * (num_classes + 1), 3))
        
        # Plot original GT
        axes[0].imshow(gt, cmap='gray')
        axes[0].set_title('Original GT')
        axes[0].axis('off')
        
        # Plot each class separately
        for i, cls in enumerate(unique_classes):
            class_mask = (gt == cls).astype(float)
            axes[i + 1].imshow(class_mask, cmap='gray')
            axes[i + 1].set_title(f'Class {cls}')
            axes[i + 1].axis('off')
        
        # Add overall title with sample info
        caption = sample['text_captions']
        if isinstance(caption, list):
            caption = ', '.join(caption)
        plt.suptitle(f"Sample {idx}: {caption}\nMean: {np.mean(gt):.3f}, Unique classes: {unique_classes}", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"gt_sample_{idx}_multiclass.png", dpi=300)
        plt.close()

    print(f"GT samples visualizations saved as 'gt_sample_X_multiclass.png'")

if __name__=='__main__':
    # Example usage with the new flags
    use_class_frequencies = False
    use_single_normalization = False
    use_color_jitter = False
    
    # dataset = ChangeDataset("train", ".png", ".png", ".png", "/mnt/store/ykorkma1/datasets/second_dataset_with_1ch/train", ['non-change', 'impervious surface', 'bare ground', 'low vegetation', 'medium vegetation','non-vegetated ground surface', 'tree', 'water bodies', 'building','playground'], 
    #                             selected_indices_file=None, use_class_frequencies=use_class_frequencies, synthetic_dataset=True, synthetic_A=False, use_single_normalization=use_single_normalization, use_color_jitter=use_color_jitter, num_samples=None, synthetic_B_folder=None, mask_B=False, missing_classes_fill=False)
    # dataset = ChangeDataset("train", ".png", ".png", ".png", "/mnt/store/ykorkma1/datasets/cnamcd_dataset_with_1ch/train", ['non-change', 'impervious surface', 'bare ground', 'low vegetation', 'medium vegetation','non-vegetated ground surface', 'tree', 'water bodies', 'building','playground'], 
    #                                 selected_indices_file=None, use_class_frequencies=use_class_frequencies, synthetic_dataset=True, synthetic_A=False, use_single_normalization=use_single_normalization, use_color_jitter=use_color_jitter, num_samples=None, synthetic_B_folder=None, mask_B=False, missing_classes_fill=False)

    dataset = ChangeDataset("train", ".png", ".png", ".png", "/mnt/store/ykorkma1/datasets/inpainted_loveda_with_finetune_val_corrected", ['non-change', 'building', 'impervious surface', 'water bodies', 'bare ground', 'tree', 'low vegetation'], 
                                use_class_frequencies=use_class_frequencies, broken_B=True, use_single_normalization=use_single_normalization, use_color_jitter=False)
    # dataset = ChangeDataset("val", ".png", ".png", ".png", "/mnt/store/ykorkma1/datasets/LEVIR-CD256", ['non-change', 'building'], 
    #                     upsample=False, use_single_normalization=use_single_normalization)
    item_a_array, item_b_array, gt_array, captions = analyze_dataset(dataset)
    
    # Visualize ground truth samples
   # visualize_gt_samples(dataset)















