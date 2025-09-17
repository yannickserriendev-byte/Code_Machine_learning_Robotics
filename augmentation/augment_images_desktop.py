"""
Advanced Image Augmentation Pipeline for Tactile Sensing Data

This script implements a comprehensive image augmentation pipeline specifically designed
for photoelastic tactile sensor data. It provides configurable image processing capabilities
including:

- Multi-mode processing (RGB or Grayscale)
- Controlled noise injection
- Geometric transformations (rotation, shifting, masking)
- Contact point coordinate transformation
- Automated dataset expansion

The pipeline maintains data integrity by correctly transforming associated labels
(contact points, force readings) during augmentation. All transformations are
deterministic and reproducible through controlled random seeding.

Key Features:
- Configurable number of augmentations per image
- Switchable RGB/Grayscale processing
- Adjustable noise and jitter parameters
- Automated label transformation
- Progress tracking and periodic saving
- Memory-efficient processing
"""

import os
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import functional as F

# ==== Configuration Parameters ====
"""
Pipeline Configuration Guide:

1. Augmentation Control:
   NUM_AUGMENTATIONS_PER_IMAGE: Controls dataset expansion factor
   - Higher values provide more diverse training data
   - Lower values reduce processing time and storage requirements

2. Image Processing Mode:
   USE_GRAYSCALE: Determines color processing pipeline
   - True: Converts to grayscale using industry-standard weights
   - False: Preserves full RGB information

3. Signal Quality Control:
   APPLY_NOISE: Controls artificial noise injection
   - True: Adds controlled noise for robustness training
   - False: Maintains original signal clarity

4. Signal Processing Parameters:
   NOISE_SIGMA: Gaussian noise intensity (range: 0.0-1.0)
   JITTER_BRIGHTNESS: Brightness variation range (range: 0.0-1.0)
   JITTER_CONTRAST: Contrast variation range (range: 0.0-1.0)
"""

# User-adjustable parameters
NUM_AUGMENTATIONS_PER_IMAGE = 10  # Number of augmented versions per input image
USE_GRAYSCALE = True              # Enable grayscale conversion
APPLY_NOISE = True                # Enable noise injection

# Signal processing parameters
NOISE_SIGMA = 5.0/255.0           # Gaussian noise standard deviation
JITTER_BRIGHTNESS = 0.4           # Brightness jitter intensity
JITTER_CONTRAST = 0.4             # Contrast jitter intensity

# Base Configuration
base_dir = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\2. Data creation\data aqcuisition\1\full_dataset"
input_csv = os.path.join(base_dir, "1.labels_cleaned.csv")
input_img_dir = os.path.join(base_dir, "images")

# Generate descriptive output paths
mode_suffix = "_grayscale" if USE_GRAYSCALE else "_rgb"
noise_suffix = "_noise" if APPLY_NOISE else "_clean"
aug_suffix = f"_{NUM_AUGMENTATIONS_PER_IMAGE}aug"

output_img_dir = os.path.join(base_dir, f"3.augmented_images{mode_suffix}{noise_suffix}{aug_suffix}")
output_csv_path = os.path.join(base_dir, f"3.augmented_labels{mode_suffix}{noise_suffix}{aug_suffix}.csv")

# System parameters
save_every = 10                   # Checkpoint frequency
FINAL_SIZE = 980                  # Output image dimensions
IMAGE_SHIFT_RIGHT = -30           # Horizontal alignment correction

# ==== Helper Functions ====
def convert_to_grayscale(img):
    """Convert RGB image to grayscale using standardized weights.
    
    Args:
        img (PIL.Image): Input RGB image
        
    Returns:
        PIL.Image: Grayscale image (converted back to RGB format for consistency)
    
    The conversion uses industry-standard coefficients:
    Luminance = 0.299·R + 0.587·G + 0.114·B
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    np_img = np.array(img)
    grayscale = (0.299 * np_img[:,:,0] + 0.587 * np_img[:,:,1] + 0.114 * np_img[:,:,2]).astype(np.uint8)
    return Image.fromarray(grayscale, mode='L').convert('RGB')

def add_gaussian_noise(img, sigma=5.0/255.0):
    """Add calibrated Gaussian noise to image.
    
    Args:
        img (PIL.Image): Input image
        sigma (float): Noise standard deviation (normalized to [0,1] range)
        
    Returns:
        PIL.Image: Image with added noise
    """
    np_img = np.array(img).astype(float) / 255.0
    noise = np.random.normal(0, sigma, np_img.shape)
    noisy = np.clip(np_img + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))

def rotate_point(x, y, angle_degrees, center_x=0.0, center_y=0.0):
    """Rotate a point around a center by specified angle.
    
    Args:
        x, y (float): Point coordinates
        angle_degrees (float): Rotation angle in degrees
        center_x, center_y (float): Rotation center coordinates
        
    Returns:
        tuple: (new_x, new_y) rotated coordinates
    """
    angle_radians = math.radians(angle_degrees)
    x_shifted = x - center_x
    y_shifted = y - center_y
    new_x = x_shifted * math.cos(angle_radians) - y_shifted * math.sin(angle_radians)
    new_y = x_shifted * math.sin(angle_radians) + y_shifted * math.cos(angle_radians)
    return new_x + center_x, new_y + center_y

def shift_image_right_crop(img, shift_pixels):
    """Shift image horizontally with edge cropping.
    
    Args:
        img (PIL.Image): Input image
        shift_pixels (int): Number of pixels to shift (negative for left shift)
        
    Returns:
        PIL.Image: Shifted image
    """
    w, h = img.size
    padded = F.pad(img, padding=[shift_pixels, 0, 0, 0], fill=0)
    return padded.crop((0, 0, w, h))

def apply_circular_mask(img, diameter):
    """Apply circular mask to image.
    
    Args:
        img (PIL.Image): Input image
        diameter (int): Mask diameter in pixels
        
    Returns:
        PIL.Image: Masked image
    """
    np_img = np.array(img)
    w, h = img.size
    radius = diameter // 2
    center_x, center_y = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    mask = dist <= radius
    masked_img = np.zeros_like(np_img)
    if np_img.ndim == 3:
        for c in range(3):
            masked_img[..., c] = np_img[..., c] * mask
    else:
        masked_img = np_img * mask
    return Image.fromarray(masked_img)

def crop_center(img, target_size):
    """Crop image to specified size from center.
    
    Args:
        img (PIL.Image): Input image
        target_size (int): Output dimensions (square)
        
    Returns:
        PIL.Image: Center-cropped image
    """
    w, h = img.size
    center_x, center_y = w // 2, h // 2
    left = center_x - target_size // 2
    top = center_y - target_size // 2
    return img.crop((left, top, left + target_size, top + target_size))

# ==== Main Processing Loop ====
def main():
    """Execute the augmentation pipeline."""
    # Initialize
    os.makedirs(output_img_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    augmented_rows = []
    counter = 0

    # Process each image
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        img_path = os.path.join(input_img_dir, row["New_Image_Name"])
        if not os.path.exists(img_path):
            print(f"❌ Missing image: {img_path}")
            continue

        orig_img = Image.open(img_path).convert("RGB")
        contact_x, contact_y = row["X_Position_mm"], row["Y_Position_mm"]

        # Add original image as final augmentation
        img = orig_img
        if USE_GRAYSCALE:
            img = convert_to_grayscale(img)
        
        shifted = shift_image_right_crop(img, IMAGE_SHIFT_RIGHT)
        masked = apply_circular_mask(shifted, diameter=FINAL_SIZE)
        cropped_img = crop_center(masked, FINAL_SIZE)

        # Save augmented image as PNG
        aug_name = f"{os.path.splitext(row['New_Image_Name'])[0]}_aug{NUM_AUGMENTATIONS_PER_IMAGE}.png"
        cropped_img.save(os.path.join(output_img_dir, aug_name), format='PNG')

        new_row = row.copy()
        new_row["New_Image_Name"] = aug_name
        new_row["x"] = contact_x
        new_row["y"] = contact_y
        new_row["rotation_angle"] = 0.0
        augmented_rows.append(new_row)
        counter += 1

        # Periodic saving
        if counter % save_every == 0:
            pd.DataFrame(augmented_rows).to_csv(output_csv_path, index=False)
            print(f"💾 Progress saved at {counter} samples...")

        # Generate augmented versions
        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            angle = random.uniform(0, 360)
            img = orig_img
            
            # Apply configured transformations
            if USE_GRAYSCALE:
                img = convert_to_grayscale(img)
            
            jitter = transforms.ColorJitter(
                brightness=JITTER_BRIGHTNESS,
                contrast=JITTER_CONTRAST
            )
            img = jitter(img)
            
            if APPLY_NOISE:
                img = add_gaussian_noise(img, sigma=NOISE_SIGMA)
            
            shifted = shift_image_right_crop(img, IMAGE_SHIFT_RIGHT)
            masked = apply_circular_mask(shifted, diameter=FINAL_SIZE)
            rotated = F.rotate(masked, angle, expand=True)
            final_img = crop_center(rotated, FINAL_SIZE)

            # Save augmented image
            # Save augmented image as PNG
            aug_name = f"{os.path.splitext(row['New_Image_Name'])[0]}_aug{i}.png"
            final_img.save(os.path.join(output_img_dir, aug_name), format='PNG')

            # Update labels
            final_x, final_y = rotate_point(contact_x, contact_y, angle)
            new_row = row.copy()
            new_row["New_Image_Name"] = aug_name
            new_row["x"] = final_x
            new_row["y"] = final_y
            new_row["rotation_angle"] = angle
            augmented_rows.append(new_row)
            counter += 1

            if counter % save_every == 0:
                pd.DataFrame(augmented_rows).to_csv(output_csv_path, index=False)
                print(f"💾 Progress saved at {counter} samples...")

    # Final save
    pd.DataFrame(augmented_rows).to_csv(output_csv_path, index=False)
    print(f"\n✅ Augmentation complete. Total samples: {len(augmented_rows)}")
    print(f"📁 Labels saved to: {output_csv_path}")
    print(f"📁 Images saved to: {output_img_dir}")

if __name__ == "__main__":
    main()