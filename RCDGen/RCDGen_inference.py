import PIL.Image
import torch
from diffusers.pipelines.stable_diffusion.RCDGenSDPipeline import StableDiffusionInstructPix2PixPipeline
from diffusers import UNet2DConditionModel
import os


def get_file_names(root_path, split_name="train"):
    """Read file names from split file."""
    assert split_name in ['train', 'val', 'test']
    source = os.path.join(root_path, split_name + '.txt')

    file_names = []
    with open(source) as f:
        files = f.readlines()

    for item in files:
        file_name = item.strip()
        if len(file_name) > 4 and file_name[-4] == '.':
            file_name = file_name[:-4]
        file_names.append(file_name)

    return file_names


def create_grid(original, post_image, mask):
    """Create a grid visualization: [original | post-image | mask]"""
    width, height = original.size
    
    # Resize all images to same size
    post_image = post_image.resize((width, height))
    mask_rgb = mask.convert("RGB").resize((width, height))
    
    # Create grid
    grid = PIL.Image.new('RGB', (width * 3, height))
    grid.paste(original, (0, 0))
    grid.paste(post_image, (width, 0))
    grid.paste(mask_rgb, (width * 2, 0))
    
    return grid


def main():
    # Model from HuggingFace Hub
    model_id = "yilmazkorkmaz/RCDGen"
    
    # Dataset path
    root_path = "/path/to/dataset"  # <- replace this
    result_dir = "outputs/images"
    result_dir_mask = "outputs/masks"
    result_dir_grid = "outputs/grids"
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir_mask, exist_ok=True)
    os.makedirs(result_dir_grid, exist_ok=True)
    
    # Load pipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")
    
    # Load EMA UNet (recommended)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet_ema", torch_dtype=torch.float16
    )
    pipe.unet = unet.cuda()
    
    generator = torch.Generator("cuda").manual_seed(42)
    
    file_names = get_file_names(root_path, split_name="train")
    
    # Change categories
    categories = ['building', 'low vegetation', 'tree', 'bare ground']  
    # customize the categories but there are only 10 categories used during training the rest will be unreliable:
    # ['non-change', 'low vegetation', 'non-vegetated ground surface', 'tree', 'water bodies', 'building', 'playground', 'medium vegetation', 'impervious surface', 'bare ground']
    image_idx = 0
    for item_name in file_names:
        image_path = os.path.join(root_path, 'A', item_name + ".png")
        image = PIL.Image.open(image_path).convert("RGB")
        
        for cat_idx, category in enumerate(categories):
            # Output paths with index and category
            filename = f"{image_idx}_{category.replace(' ', '_')}"
            post_image_path = os.path.join(result_dir, f"{filename}.png")
            mask_path = os.path.join(result_dir_mask, f"{filename}.png")
            grid_path = os.path.join(result_dir_grid, f"{filename}.png")
            
            # Skip if already exists
            if os.path.exists(post_image_path) and os.path.exists(mask_path):
                print(f"Skipping {filename} (already exists)")
                image_idx += 1
                continue
            
            # Generate
            prompt = f"change in {category}"
            output = pipe(
                prompt,
                image=image,
                num_inference_steps=100,
                image_guidance_scale=1.5,
                guidance_scale=7.0,
                generator=generator,
            ).images
            
            post_image = output[0][0]
            mask = output[1][0]
            
            # Save individual files
            post_image.save(post_image_path)
            mask.save(mask_path)
            
            # Save grid visualization
            grid = create_grid(image, post_image, mask)
            grid.save(grid_path)
            
            print(f"Saved {image_idx}: {item_name} - {category}")
            image_idx += 1


if __name__ == "__main__":
    main()
