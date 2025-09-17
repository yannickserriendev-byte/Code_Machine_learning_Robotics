"""
Augmentation utility for Sensor 1 images.

- Augments each im# ==== Prepare folders ====
os.makedirs(output_img_dir, exist_ok=True)

# ==== Load CSV ====
df = pd.read_csv(input_csv)
augmented_rows = []

# ==== Helper functions ====nfigurable number of rotations, color jitter, and circular masking.
- Configurable: num_augmentations_per_image, FINAL_SIZE, IMAGE_SHIFT_RIGHT.
- Outputs augmented images and updated labels CSV with rotated contact points.
- Adjust base_dir, input_csv, and input_img_dir for your dataset location.
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

# ==== User Configuration ====
"""
Augmentation Configuration Parameters:

1. Number of Augmentations:
   - Thesis best results: 10 for grayscale, 3 for RGB
   - Set NUM_AUGMENTATIONS_PER_IMAGE to your desired number

2. Image Mode:
   - Set USE_GRAYSCALE = True for grayscale conversion
   - Set USE_GRAYSCALE = False for RGB

3. Noise Application:
   - Set APPLY_NOISE = True to add Gaussian noise
   - Set APPLY_NOISE = False for clean images

4. Noise Parameters (if APPLY_NOISE is True):
   - NOISE_SIGMA: Standard deviation for Gaussian noise (thesis: 5.0/255.0)
   - JITTER_BRIGHTNESS: Brightness jitter intensity (thesis: 0.4)
   - JITTER_CONTRAST: Contrast jitter intensity (thesis: 0.4)
"""

# User-adjustable parameters
NUM_AUGMENTATIONS_PER_IMAGE = 10  # Change this to your desired number of augmentations
USE_GRAYSCALE = True              # Set False for RGB images
APPLY_NOISE = True                # Set False for clean images without noise

# Noise parameters (based on thesis optimal values)
NOISE_SIGMA = 5.0/255.0           # Gaussian noise standard deviation
JITTER_BRIGHTNESS = 0.4           # Brightness jitter intensity
JITTER_CONTRAST = 0.4             # Contrast jitter intensity

# Base Configuration
base_dir = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\2. Data creation\data aqcuisition\1\full_dataset"
input_csv = os.path.join(base_dir, "1.labels_cleaned.csv")
input_img_dir = os.path.join(base_dir, "images")

# Create descriptive suffix for output files based on settings
mode_suffix = "_grayscale" if USE_GRAYSCALE else "_rgb"
noise_suffix = "_noise" if APPLY_NOISE else "_clean"
aug_suffix = f"_{NUM_AUGMENTATIONS_PER_IMAGE}aug"

output_img_dir = os.path.join(base_dir, f"3.augmented_images{mode_suffix}{noise_suffix}{aug_suffix}")
output_csv_path = os.path.join(base_dir, f"3.augmented_labels{mode_suffix}{noise_suffix}{aug_suffix}.csv")

save_every = 10
FINAL_SIZE = 980  # Thesis specifies 980x980 pixels for Sensor 1
IMAGE_SHIFT_RIGHT = -30  # Horizontal shift to correct alignment

# ==== Prepare folders ====
os.makedirs(output_img_dir, exist_ok=True)

# ==== Load CSV ====
df = pd.read_csv(input_csv)
augmented_rows = []

# ==== Augmentations ====
color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4)  # Remove saturation and hue as we'll convert to grayscale

# ==== Helper functions ====
def convert_to_grayscale(img):
    """Convert RGB image to grayscale using thesis-specified weights:
    Grayscale = 0.299·Red + 0.587·Green + 0.114·Blue
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    np_img = np.array(img)
    grayscale = (0.299 * np_img[:,:,0] + 0.587 * np_img[:,:,1] + 0.114 * np_img[:,:,2]).astype(np.uint8)
    return Image.fromarray(grayscale, mode='L').convert('RGB')  # Convert back to RGB for model compatibility

def add_gaussian_noise(img, sigma=5.0/255.0):
    """Add Gaussian noise to image with thesis-specified sigma value.
    Args:
        img: PIL Image
        sigma: Standard deviation of the noise (default: 5.0/255.0 as specified in thesis)
    Returns:
        PIL Image with added noise
    """
    np_img = np.array(img).astype(float) / 255.0  # Normalize to [0,1]
    noise = np.random.normal(0, sigma, np_img.shape)
    noisy = np.clip(np_img + noise, 0, 1)  # Ensure values stay in [0,1]
    return Image.fromarray((noisy * 255).astype(np.uint8))
def rotate_point(x, y, angle_degrees, center_x=0.0, center_y=0.0):
    angle_radians = math.radians(angle_degrees)
    x_shifted = x - center_x
    y_shifted = y - center_y
    new_x = x_shifted * math.cos(angle_radians) - y_shifted * math.sin(angle_radians)
    new_y = x_shifted * math.sin(angle_radians) + y_shifted * math.cos(angle_radians)
    return new_x + center_x, new_y + center_y

def shift_image_right_crop(img, shift_pixels):
    w, h = img.size
    padded = F.pad(img, padding=[shift_pixels, 0, 0, 0], fill=0)
    return padded.crop((0, 0, w, h))

def apply_circular_mask(img, diameter):
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
    w, h = img.size
    center_x, center_y = w // 2, h // 2
    left = center_x - target_size // 2
    top = center_y - target_size // 2
    return img.crop((left, top, left + target_size, top + target_size))

# ==== Augmentation Loop ====
counter = 0
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):

    img_path = os.path.join(input_img_dir, row["New_Image_Name"])
    if not os.path.exists(img_path):
        print(f"❌ Missing image: {img_path}")
        continue

    orig_img = Image.open(img_path).convert("RGB")
    contact_x, contact_y = row["X_Position_mm"], row["Y_Position_mm"]

    # === Add original image as final augmentation ===
    # Process original image based on user configuration
    img = orig_img
    if USE_GRAYSCALE:
        img = convert_to_grayscale(img)
    
    # Apply basic transformations (always needed)
    shifted = shift_image_right_crop(img, IMAGE_SHIFT_RIGHT)
    masked = apply_circular_mask(shifted, diameter=FINAL_SIZE)
    cropped_img = crop_center(masked, FINAL_SIZE)

    aug_name = f"{os.path.splitext(row['New_Image_Name'])[0]}_aug{NUM_AUGMENTATIONS_PER_IMAGE}.png"
    cropped_img.save(os.path.join(output_img_dir, aug_name))

    new_row = row.copy()
    new_row["New_Image_Name"] = aug_name
    new_row["x"] = contact_x
    new_row["y"] = contact_y
    new_row["rotation_angle"] = 0.0
    augmented_rows.append(new_row)
    counter += 1

    if counter % save_every == 0:
        pd.DataFrame(augmented_rows).to_csv(output_csv_path, index=False)
        print(f"💾 Progress saved at {counter} samples...")

    # === Generate augmented images ===
    for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
        angle = random.uniform(0, 360)
        img = orig_img
        
        # 1. Convert to grayscale if specified
        if USE_GRAYSCALE:
            img = convert_to_grayscale(img)
        
        # 2. Apply brightness/contrast jitter (with user-defined parameters)
        jitter = transforms.ColorJitter(
            brightness=JITTER_BRIGHTNESS,
            contrast=JITTER_CONTRAST
        )
        img = jitter(img)
        
        # 3. Add Gaussian noise if specified
        if APPLY_NOISE:
            img = add_gaussian_noise(img, sigma=NOISE_SIGMA)
        
        # 4. Shift image to correct alignment
        shifted = shift_image_right_crop(img, IMAGE_SHIFT_RIGHT)
        
        # 5. Apply circular mask
        masked = apply_circular_mask(shifted, diameter=FINAL_SIZE)
        
        # 6. Rotate around center
        rotated = F.rotate(masked, angle, expand=True)
        
        # 7. Final center crop
        final_img = crop_center(rotated, FINAL_SIZE)

        # Rotate contact point around sensor-centered origin (0, 0)
        final_x, final_y = rotate_point(contact_x, contact_y, angle)

        aug_name = f"{os.path.splitext(row['New_Image_Name'])[0]}_aug{i}.png"
        final_img.save(os.path.join(output_img_dir, aug_name))

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

# ==== Final Save ====
pd.DataFrame(augmented_rows).to_csv(output_csv_path, index=False)
print(f"\n✅ Finished. Total augmented + original samples: {len(augmented_rows)}")
print(f"📁 Labels saved to: {output_csv_path}")
print(f"📁 Images saved to: {output_img_dir}")
