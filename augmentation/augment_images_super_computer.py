"""
High-Performance Image Augmentation Pipeline for Tactile Sensing Data

This script implements a parallel-processing optimized augmentation pipeline designed
for high-performance computing environments. It uses tensor-based operations and
multi-threading for efficient processing of large datasets.

Key Features:
- Parallel processing with configurable thread count
- Tensor-based image operations for improved performance
- Memory-efficient data handling
- Configurable augmentation parameters
- Thread-safe error handling
- Optimized for high-throughput processing
"""

import os
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==== Configuration Parameters ====
"""
HPC Pipeline Configuration:

1. Processing Control:
   NUM_THREADS: Number of parallel processing threads
   - Higher values increase throughput on multi-core systems
   - Recommended: Number of CPU cores - 1

2. Augmentation Parameters:
   NUM_AUGMENTATIONS_PER_IMAGE: Dataset expansion factor
   USE_GRAYSCALE: Color processing mode
   APPLY_NOISE: Noise injection control

3. Signal Processing:
   NOISE_SIGMA: Noise intensity (normalized)
   JITTER_BRIGHTNESS: Brightness variation
   JITTER_CONTRAST: Contrast variation
"""

# Processing parameters
NUM_THREADS = 47                  # Adjust based on available CPU cores
BATCH_SIZE = 32                   # Number of images per batch

# Augmentation parameters
NUM_AUGMENTATIONS_PER_IMAGE = 10  # Number of augmented versions per input image
USE_GRAYSCALE = True              # Enable grayscale conversion
APPLY_NOISE = True                # Enable noise injection

# Signal processing parameters
NOISE_SIGMA = 5.0/255.0           # Gaussian noise standard deviation
JITTER_BRIGHTNESS = 0.4           # Brightness jitter intensity
JITTER_CONTRAST = 0.4             # Contrast jitter intensity

# System parameters
FINAL_SIZE = 980                  # Output image dimensions
IMAGE_SHIFT_RIGHT = -30           # Horizontal alignment correction

# Base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
COMBINED_DATASET = os.path.join(PROJECT_ROOT, "data", "combined_dataset")

# Input files from combined dataset
input_csv = os.path.join(COMBINED_DATASET, "labels.csv")  # Combined labels from preprocessing
input_img_dir = os.path.join(COMBINED_DATASET, "images")  # Combined images from preprocessing

# Generate output paths
mode_suffix = "_grayscale" if USE_GRAYSCALE else "_rgb"
noise_suffix = "_noise" if APPLY_NOISE else "_clean"
aug_suffix = f"_{NUM_AUGMENTATIONS_PER_IMAGE}aug"
thread_suffix = f"_{NUM_THREADS}threads"

# Create output directories relative to project root
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "augmented_dataset")
output_img_dir = os.path.join(OUTPUT_DIR, f"images{mode_suffix}{noise_suffix}{aug_suffix}{thread_suffix}")
output_csv_path = os.path.join(OUTPUT_DIR, f"labels{mode_suffix}{noise_suffix}{aug_suffix}{thread_suffix}.csv")

# ==== Tensor Operations ====
def add_gaussian_noise_tensor(tensor: torch.Tensor, sigma: float = NOISE_SIGMA) -> torch.Tensor:
    """Add Gaussian noise to tensor in [0,1] range."""
    noise = torch.randn_like(tensor) * sigma
    return torch.clamp(tensor + noise, 0.0, 1.0)

def shift_image_tensor(tensor: torch.Tensor, shift_pixels: int) -> torch.Tensor:
    """Shift tensor horizontally with edge handling."""
    pad_left = max(shift_pixels, 0)
    pad_right = max(-shift_pixels, 0)
    padded = F.pad(tensor, [pad_left, 0, pad_right, 0], fill=0.0)
    return padded[:, :tensor.shape[1], :tensor.shape[2]]

def apply_circular_mask_tensor(tensor: torch.Tensor, diameter: int) -> torch.Tensor:
    """Apply circular mask to tensor."""
    _, H, W = tensor.shape
    radius = diameter // 2
    cx, cy = W // 2, H // 2
    y = torch.arange(H, device=tensor.device).view(H, 1).expand(H, W)
    x = torch.arange(W, device=tensor.device).view(1, W).expand(H, W)
    dist = torch.sqrt((x - cx).float()**2 + (y - cy).float()**2)
    mask = (dist <= radius).float()
    return tensor * mask.unsqueeze(0)

def rotate_point(x: float, y: float, angle_degrees: float, 
                center_x: float = 0.0, center_y: float = 0.0) -> tuple:
    """Rotate point coordinates."""
    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan
    theta = math.radians(angle_degrees)
    xs, ys = x - center_x, y - center_y
    nx = xs*math.cos(theta) - ys*math.sin(theta) + center_x
    ny = xs*math.sin(theta) + ys*math.cos(theta) + center_y
    return nx, ny

# ==== Image Processing Function ====
def process_row(row: pd.Series, output_folder: str) -> list:
    """Process single image with all augmentations in separate thread."""
    results = []
    # Use filename from combined dataset
    img_filename = row["New_Image_Name"]
    stem = os.path.splitext(img_filename)[0]
    img_path = os.path.join(input_img_dir, img_filename)

    try:
        # Check if TIFF file exists
        if not os.path.exists(img_path):
            print(f"❌ Missing TIFF image: {img_path}")
            return []

        # Convert to tensor with appropriate color mode
        if USE_GRAYSCALE:
            base_tensor = transforms.ToTensor()(
                Image.open(img_path).convert("L")
            ).repeat(3, 1, 1)
        else:
            base_tensor = transforms.ToTensor()(
                Image.open(img_path).convert("RGB")
            )

        # Get contact position from position data
        contact_x = row["X_Position_mm"]  # Using actual position from measurement setup
        contact_y = row["Y_Position_mm"]  # Using actual position from measurement setup

        # Generate augmentations
        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            angle = random.uniform(0, 360)
            t = base_tensor.clone()

            # Apply transformations
            if APPLY_NOISE:
                t = add_gaussian_noise_tensor(t)

            # Apply color jitter
            pil = transforms.ToPILImage()(t)
            pil = transforms.ColorJitter(
                brightness=JITTER_BRIGHTNESS,
                contrast=JITTER_CONTRAST
            )(pil)
            t = transforms.ToTensor()(pil)

            # Apply geometric transformations
            t = shift_image_tensor(t, IMAGE_SHIFT_RIGHT)
            t = apply_circular_mask_tensor(t, diameter=FINAL_SIZE)
            t = F.rotate(t, angle, expand=True)
            t = F.center_crop(t, [FINAL_SIZE, FINAL_SIZE])

            # Save augmented image
            out_pil = transforms.ToPILImage()(t)
            aug_name = f"{stem}_aug{i}.png"
            out_path = os.path.join(output_folder, aug_name)
            out_pil.save(out_path, format="PNG")

            # Update metadata
            final_x, final_y = rotate_point(contact_x, contact_y, angle)
            meta = row.copy()
            meta["New_Image_Name"] = aug_name
            meta["x"] = final_x
            meta["y"] = final_y
            meta["rotation_angle"] = angle
            results.append(meta)

    except Exception as e:
        print(f"❌ Error processing {img_filename}: {str(e)}")
        return []

    return results

# ==== Main Processing Loop ====
def main():
    """Execute parallel augmentation pipeline."""
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Load pre-merged dataset from combination script
    df_full = pd.read_csv(input_csv)
    
    # Verify all expected images exist
    expected_images = df_full["New_Image_Name"].tolist()
    missing_images = [img for img in expected_images if not os.path.exists(os.path.join(input_img_dir, img))]
    if missing_images:
        print(f"⚠️ Warning: {len(missing_images)} images missing from {input_img_dir}")
        for img in missing_images[:5]:  # Show first 5 missing
            print(f"  Missing: {img}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")
    
    # Use only rows where images exist
    df = df_full[df_full["Image_Number"].apply(
        lambda x: os.path.exists(os.path.join(input_img_dir, f"image_{int(x)}.tiff"))
    )].copy()
    
    # Initialize result dataframe with all needed columns
    result_columns = list(df.columns) + ["x", "y", "rotation_angle", "augmentation_id"]
    df_aug = pd.DataFrame(columns=result_columns)

    # Process images in parallel
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_row, row, output_img_dir)
            for _, row in df.iterrows()
        ]
        
        # Process results as they complete
        for future in tqdm(
            as_completed(futures), 
            total=len(futures), 
            desc=f"Augmenting with {NUM_THREADS} threads"
        ):
            df_aug = pd.concat(
                [df_aug, pd.DataFrame(future.result())],
                ignore_index=True
            )

    # Save final results
    df_aug.to_csv(output_csv_path, index=False)
    print(f"\n✅ Augmentation complete. Generated {len(df_aug)} samples")
    print(f"📁 Labels saved to: {output_csv_path}")
    print(f"📁 Images saved to: {output_img_dir}")

if __name__ == "__main__":
    main()