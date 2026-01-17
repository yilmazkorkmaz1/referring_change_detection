"""
Utility script to convert RGB-encoded label images to single-channel class index images.
Maps specific RGB colors to class indices for change detection datasets.
"""
import os
import argparse
from PIL import Image
import numpy as np


def convert_rgb_to_class(input_folder: str, output_folder: str, color_to_class: dict):
    """
    Convert RGB-encoded labels to class index images.
    
    Args:
        input_folder: Path to folder containing RGB label images
        output_folder: Path to save converted class images
        color_to_class: Dictionary mapping RGB tuples to class indices
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each image in the folder
    processed_count = 0
    for filename in os.listdir(input_folder):
        if not (filename.endswith('.png') or filename.endswith('.jpg')):
            continue
            
        # Open the image
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure it's in RGB mode
        
        # Convert image to numpy array for processing
        image_array = np.array(image)
        
        # Create an empty array for the class image
        class_image_array = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
        
        # Map colors to class values
        for color, class_value in color_to_class.items():
            mask = np.all(image_array == color, axis=-1)
            class_image_array[mask] = class_value
        
        # Check for unmapped pixels
        mapped_pixels = np.zeros_like(class_image_array, dtype=bool)
        for color in color_to_class.keys():
            mapped_pixels |= np.all(image_array == color, axis=-1)
        
        unmapped_count = (~mapped_pixels).sum()
        if unmapped_count > 0:
            unique_colors = np.unique(image_array.reshape(-1, 3), axis=0)
            unmapped_colors = [tuple(c) for c in unique_colors 
                             if tuple(c) not in color_to_class]
            print(f"Warning: {filename} has {unmapped_count} unmapped pixels")
            print(f"  Unmapped colors: {unmapped_colors}")
        
        # Convert the numpy array back to an image
        class_image = Image.fromarray(class_image_array, mode='L')  # 'L' mode for grayscale
        
        # Save the class image
        output_path = os.path.join(output_folder, filename)
        class_image.save(output_path)
        processed_count += 1
    
    print(f"Processed {processed_count} images and saved to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RGB labels to class index images")
    parser.add_argument("--input", "-i", type=str, 
                       default="data/second_dataset/test/label2",
                       help="Input folder containing RGB label images")
    parser.add_argument("--output", "-o", type=str,
                       default="data/second_dataset/test/gt",
                       help="Output folder for class index images")
    args = parser.parse_args()

    # Define the color-to-class mapping for SECOND dataset
    # Modify this mapping based on your dataset's color scheme
    color_to_class = {
        (255, 255, 255): 0,  # Non-change (White)
        (0, 128, 0): 1,      # Low Vegetation (Dark Green)
        (128, 128, 128): 2,  # Non-vegetated Ground Surface (Gray)
        (0, 255, 0): 3,      # Tree (Green)
        (0, 0, 255): 4,      # Water (Blue)
        (128, 0, 0): 5,      # Building (Dark Red)
        (255, 0, 0): 6       # Playground (Red)
    }
    
    convert_rgb_to_class(args.input, args.output, color_to_class)
