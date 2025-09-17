"""
Advanced Image Augmentation Pipeline for Tactile Sensing Data

This script implements a comprehensive image augmentation pipeline with configurable
processing options and robust data handling to prevent CSV corruption.
"""

import os
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import functional as F

# ==== Configuration Parameters ====
# Processing Mode
USE_GRAYSCALE = False              # Set to True for grayscale conversion
APPLY_NOISE = False                 # Set to True to add Gaussian noise
NUM_AUGMENTATIONS_PER_IMAGE = 1    # Number of augmented versions per image

# Signal Processing Parameters
NOISE_SIGMA = 5.0/255.0            # Noise intensity (normalized)
JITTER_BRIGHTNESS = 0.4            # Brightness variation range
JITTER_CONTRAST = 0.4              # Contrast variation range
JITTER_SATURATION = 0.0            # Saturation variation (RGB only)
JITTER_HUE = 0.0                   # Hue variation (RGB only)

# System Parameters
FINAL_SIZE = 1200                  # Output image dimensions
IMAGE_SHIFT_RIGHT = -30            # Horizontal alignment correction
SAVE_EVERY = 10                    # Progress save frequency

# ==== Path Configuration ====
base_dir = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\code\code full pipeline\All code\Code from laptop\Testing_data_del\Data\full_dataset"
input_csv = os.path.join(base_dir, "0.labels.csv")
input_img_dir = os.path.join(base_dir, "0.images")

# Setup output paths with simple timestamp
timestamp = time.strftime("%m%d_%H%M")  # Simple timestamp format: MMDD_HHMM
output_img_dir = os.path.join(base_dir, f"1.augmented_images_{timestamp}")
output_csv_path = os.path.join(base_dir, f"1.augmented_labels_{timestamp}.csv")
working_csv = os.path.join(base_dir, "working.csv")

# Create configuration log file with detailed settings
config_content = f"""Augmentation Configuration Details
Created: {time.strftime("%Y-%m-%d %H:%M:%S")}
================================
Timestamp: {timestamp}

Processing Mode:
- Grayscale Mode: {'Enabled' if USE_GRAYSCALE else 'Disabled'}
- Noise Injection: {'Enabled' if APPLY_NOISE else 'Disabled'}
- Augmentations per Image: {NUM_AUGMENTATIONS_PER_IMAGE}

{'''Signal Processing Parameters:
- Noise Sigma: {NOISE_SIGMA}
- Brightness Jitter: {JITTER_BRIGHTNESS}
- Contrast Jitter: {JITTER_CONTRAST}
- Saturation Jitter: {JITTER_SATURATION}
- Hue Jitter: {JITTER_HUE}
''' if APPLY_NOISE else ''}
System Parameters:
- Final Image Size: {FINAL_SIZE}
- Image Shift Right: {IMAGE_SHIFT_RIGHT}
- Save Frequency: Every {SAVE_EVERY} samples

Input/Output Paths:
- Input CSV: {input_csv}
- Input Images: {input_img_dir}
- Output Images: {output_img_dir}
- Output Labels: {output_csv_path}
"""

config_path = os.path.join(base_dir, f"augmentation_config_{timestamp}.txt")
with open(config_path, 'w') as f:
    f.write(config_content)

# ==== Helper Functions ====
def safe_save_dataframe(df, working_file, final_file=None):
    """Safely save DataFrame to CSV with verification.
    
    Args:
        df: pandas DataFrame to save
        working_file: Path to the working CSV file
        final_file: Optional path for the final CSV location
    """
    try:
        # First save to working file
        df.to_csv(working_file, index=False)
        
        # Verify the file was written correctly
        try:
            pd.read_csv(working_file)
        except Exception as e:
            print(f"Error verifying CSV: {str(e)}")
            return False
            
        # If a final destination is specified and verification passed
        if final_file:
            try:
                # Copy to final destination
                with open(working_file, 'r') as source:
                    with open(final_file, 'w') as target:
                        target.write(source.read())
                        
                # Verify the final file
                pd.read_csv(final_file)
                return True
                
            except Exception as e:
                print(f"Error creating final CSV: {str(e)}")
                return False
        
        return True
            
    except Exception as e:
        print(f"Error in save operation: {str(e)}")
        return False

def convert_to_grayscale(img):
    """Convert RGB image to grayscale using standardized weights."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    np_img = np.array(img)
    grayscale = (0.299 * np_img[:,:,0] + 0.587 * np_img[:,:,1] + 0.114 * np_img[:,:,2]).astype(np.uint8)
    return Image.fromarray(grayscale, mode='L').convert('RGB')

def add_gaussian_noise(img, sigma=NOISE_SIGMA):
    """Add calibrated Gaussian noise to image."""
    np_img = np.array(img).astype(float) / 255.0
    noise = np.random.normal(0, sigma, np_img.shape)
    noisy = np.clip(np_img + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))

def rotate_point(x, y, angle_degrees, center_x=0.0, center_y=0.0):
    """Rotate a point around a center by specified angle."""
    angle_radians = math.radians(angle_degrees)
    x_shifted = x - center_x
    y_shifted = y - center_y
    new_x = x_shifted * math.cos(angle_radians) - y_shifted * math.sin(angle_radians)
    new_y = x_shifted * math.sin(angle_radians) + y_shifted * math.cos(angle_radians)
    return new_x + center_x, new_y + center_y

def shift_image_right_crop(img, shift_pixels):
    """Shift image horizontally with edge cropping."""
    w, h = img.size
    padded = F.pad(img, padding=[shift_pixels, 0, 0, 0], fill=0)
    return padded.crop((0, 0, w, h))

def apply_circular_mask(img, diameter):
    """Apply circular mask to image."""
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
    """Crop image to specified size from center."""
    w, h = img.size
    center_x, center_y = w // 2, h // 2
    left = center_x - target_size // 2
    top = center_y - target_size // 2
    return img.crop((left, top, left + target_size, top + target_size))

def process_image(img, apply_jitter=True):
    """Apply configured image processing pipeline."""
    if USE_GRAYSCALE:
        img = convert_to_grayscale(img)
    
    if apply_jitter:
        jitter = transforms.ColorJitter(
            brightness=JITTER_BRIGHTNESS,
            contrast=JITTER_CONTRAST,
            saturation=JITTER_SATURATION if not USE_GRAYSCALE else 0,
            hue=JITTER_HUE if not USE_GRAYSCALE else 0
        )
        img = jitter(img)
    
    if APPLY_NOISE:
        img = add_gaussian_noise(img, NOISE_SIGMA)
        
    return img

def main():
    """Main execution function with error handling and data validation."""
    try:
        # ==== Prepare Output Directory ====
        os.makedirs(output_img_dir, exist_ok=True)

        # ==== Load and Validate Input Data ====
        print(f"Loading data from: {input_csv}")
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        
        df = pd.read_csv(input_csv)
        required_columns = ["New_Image_Name", "X_Position_mm", "Y_Position_mm"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Initialize storage
        augmented_rows = []
        counter = 0
        last_save_counter = 0

        # ==== Augmentation Loop ====
        with tqdm(total=len(df) * (NUM_AUGMENTATIONS_PER_IMAGE + 1), desc="Augmenting") as pbar:
            for idx, row in df.iterrows():
                try:
                    img_path = os.path.join(input_img_dir, row["New_Image_Name"])
                    if not os.path.exists(img_path):
                        print(f"❌ Missing image: {img_path}")
                        continue

                    orig_img = Image.open(img_path).convert("RGB")
                    contact_x, contact_y = row["X_Position_mm"], row["Y_Position_mm"]

                    # === Add original image as final augmentation ===
                    img = process_image(orig_img, apply_jitter=False)  # No jitter for original
                    shifted = shift_image_right_crop(img, IMAGE_SHIFT_RIGHT)
                    masked = apply_circular_mask(shifted, diameter=FINAL_SIZE)
                    cropped_img = crop_center(masked, FINAL_SIZE)

                    aug_name = f"{Path(row['New_Image_Name']).stem}_aug{NUM_AUGMENTATIONS_PER_IMAGE}.png"
                    cropped_img.save(os.path.join(output_img_dir, aug_name))

                    new_row = row.copy()
                    new_row["New_Image_Name"] = aug_name
                    new_row["x"] = contact_x
                    new_row["y"] = contact_y
                    new_row["rotation_angle"] = 0.0
                    augmented_rows.append(new_row)
                    counter += 1
                    pbar.update(1)

                    # Save progress periodically
                    if counter >= last_save_counter + SAVE_EVERY:
                        if safe_save_dataframe(pd.DataFrame(augmented_rows), working_csv):
                            print(f"💾 Progress saved at {counter} samples...")
                            last_save_counter = counter

                    # === Generate augmented versions ===
                    for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
                        angle = random.uniform(0, 360)
                        
                        # Process augmented image
                        img = process_image(orig_img, apply_jitter=True)
                        shifted = shift_image_right_crop(img, IMAGE_SHIFT_RIGHT)
                        masked = apply_circular_mask(shifted, diameter=FINAL_SIZE)
                        rotated = F.rotate(masked, angle, expand=True)
                        final_img = crop_center(rotated, FINAL_SIZE)

                        # Save augmented image
                        aug_name = f"{Path(row['New_Image_Name']).stem}_aug{i}.png"
                        output_path = os.path.join(output_img_dir, aug_name)
                        final_img.save(output_path)

                        # Update metadata
                        final_x, final_y = rotate_point(contact_x, contact_y, angle)
                        new_row = row.copy()
                        new_row["New_Image_Name"] = aug_name
                        new_row["x"] = final_x
                        new_row["y"] = final_y
                        new_row["rotation_angle"] = angle
                        augmented_rows.append(new_row)
                        counter += 1
                        pbar.update(1)

                        # Save progress periodically
                        if counter >= last_save_counter + SAVE_EVERY:
                            if safe_save_dataframe(pd.DataFrame(augmented_rows), working_csv):
                                print(f"💾 Progress saved at {counter} samples...")
                                last_save_counter = counter

                except Exception as e:
                    print(f"❌ Error processing image {row['New_Image_Name']}: {str(e)}")
                    continue

        # ==== Final Save ====
        final_df = pd.DataFrame(augmented_rows)
        if safe_save_dataframe(final_df, working_csv, output_csv_path):
            print(f"\n✅ Finished. Total augmented + original samples: {len(augmented_rows)}")
            print(f"📁 Labels saved to: {output_csv_path}")
            print(f"📁 Images saved to: {output_img_dir}")
            print(f"📁 Configuration saved to: {config_path}")
        else:
            print("❌ Error saving final CSV file")

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        if os.path.exists(working_csv):
            print(f"Working CSV file available at: {working_csv}")

if __name__ == "__main__":
    main()