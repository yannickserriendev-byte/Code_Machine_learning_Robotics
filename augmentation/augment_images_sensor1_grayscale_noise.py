"""
Grayscale + Noise Augmentation Pipeline for Sensor 1 (Crisp Images)

This script performs grayscale conversion, brightness/contrast jitter, Gaussian noise addition, image shifting, circular masking, random rotation, and center cropping for Sensor 1 images. It is CPU-only and resume-capable. 

Configuration:
- Set `base_dir`, `input_csv`, `input_img_dir`, `output_dir`, and `output_csv_path` for your dataset.
- Adjust `num_augmentations_per_image`, `FINAL_SIZE`, `IMAGE_SHIFT_RIGHT`, and `num_threads` as needed.

Usage:
- Place raw images and labels CSV in the specified input folders.
- Run the script to generate augmented images and a new labels CSV.
- All steps are commented for clarity.
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

# ==== Configuration: Paths & Parameters ====
base_dir = "<SET THIS TO YOUR DATASET ROOT>"
input_csv       = os.path.join(base_dir, "0.labels.csv")
input_img_dir   = os.path.join(base_dir, "0.images")
output_dir      = os.path.join(base_dir, "1.augmented_images_full_pipeline")
output_csv_path = os.path.join(base_dir, "1.augmented_labels_full_pipeline.csv")

num_augmentations_per_image = 10
FINAL_SIZE                 = 980
IMAGE_SHIFT_RIGHT          = -30
num_threads                = 8

os.makedirs(output_dir, exist_ok=True)
device = torch.device("cpu")

# ==== Load CSV ====
df_full = pd.read_csv(input_csv)
available_images = sorted(os.listdir(input_img_dir))[:]
df = df_full[df_full["New_Image_Name"].isin(available_images)].copy()

# ==== Transforms & Helpers ====
color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4)

def add_gaussian_noise_tensor(tensor: torch.Tensor, sigma: float = 5.0/255.0) -> torch.Tensor:
    """Add Gaussian noise to a C×H×W tensor in [0,1]."""
    noise = torch.randn_like(tensor) * sigma
    return torch.clamp(tensor + noise, 0.0, 1.0)

def shift_image_tensor(tensor: torch.Tensor, shift_pixels: int) -> torch.Tensor:
    pad_left  = max(shift_pixels, 0)
    pad_right = max(-shift_pixels, 0)
    padded = F.pad(tensor, [pad_left, 0, pad_right, 0], fill=0.0)
    return padded[:, :tensor.shape[1], :tensor.shape[2]]

def apply_circular_mask_tensor(tensor: torch.Tensor, diameter: int) -> torch.Tensor:
    _, H, W = tensor.shape
    radius = diameter // 2
    cx, cy = W // 2, H // 2
    y = torch.arange(H, device=tensor.device).view(H, 1).expand(H, W)
    x = torch.arange(W, device=tensor.device).view(1, W).expand(H, W)
    dist = torch.sqrt((x - cx).float()**2 + (y - cy).float()**2)
    mask = (dist <= radius).float()
    return tensor * mask.unsqueeze(0)

def rotate_point(x, y, angle_degrees, center_x=0.0, center_y=0.0):
    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan
    theta = math.radians(angle_degrees)
    xs, ys = x - center_x, y - center_y
    nx = xs*math.cos(theta) - ys*math.sin(theta) + center_x
    ny = xs*math.sin(theta) + ys*math.cos(theta) + center_y
    return nx, ny

# ==== Per-row augmentation function ====
def process_row(row: pd.Series, output_folder: str) -> list:
    results = []
    img_filename = row["New_Image_Name"]
    stem = os.path.splitext(img_filename)[0]
    img_path = os.path.join(input_img_dir, img_filename)

    if not os.path.exists(img_path):
        print(f"❌ Missing image: {img_path}")
        return []

    # load and convert to 3-channel tensor in [0,1]
    pil_gray = Image.open(img_path).convert("L")
    base_tensor = transforms.ToTensor()(pil_gray).repeat(3,1,1)

    # original contact point
    contact_x, contact_y = row["X_Position_mm"], row["Y_Position_mm"]

    for i in range(num_augmentations_per_image):
        angle = random.uniform(0, 360)

        # 1) Brightness/Contrast jitter
        pil = transforms.ToPILImage()(base_tensor)
        pil = color_jitter(pil)
        t = transforms.ToTensor()(pil)

        # 2) Gaussian noise
        t = add_gaussian_noise_tensor(t)

        # 3) Shift
        t = shift_image_tensor(t, IMAGE_SHIFT_RIGHT)

        # 4) Circular mask
        t = apply_circular_mask_tensor(t, diameter=FINAL_SIZE)

        # 5) Rotate (around center)
        t = F.rotate(t, angle, expand=True)

        # 6) Center-crop back to FINAL_SIZE
        t = F.center_crop(t, [FINAL_SIZE, FINAL_SIZE])

        # convert to PIL and save into folder
        out_pil = transforms.ToPILImage()(t)
        aug_name = f"{stem}_aug{i}.png"
        out_path = os.path.join(output_folder, aug_name)
        out_pil.save(out_path, format="PNG")

        final_x, final_y = rotate_point(contact_x, contact_y, angle)

        meta = row.copy()
        meta["New_Image_Name"] = aug_name
        meta["x"] = final_x
        meta["y"] = final_y
        meta["rotation_angle"] = angle

        results.append(meta)

    return results

# ==== Main ====
def main():
    df_aug = pd.DataFrame(columns=list(df.columns) + ["x","y","rotation_angle"])

    with ThreadPoolExecutor(max_workers=num_threads) as exe:
        futures = [exe.submit(process_row, row, output_dir) for _, row in df.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting"):
            df_aug = pd.concat([df_aug, pd.DataFrame(future.result())], ignore_index=True)

    # Save augmented labels CSV
    df_aug.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df_aug)} augmented images and labels to {output_dir} and {output_csv_path}")

if __name__ == "__main__":
    main()
