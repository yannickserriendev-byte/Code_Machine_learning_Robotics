"""
Augmentation utility for Sensor 1 images.

- Augments each image with configurable number of rotations, color jitter, and circular masking.
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

# ==== Configuration ====
base_dir = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\2. Data creation\data aqcuisition\1\full_dataset"
input_csv = os.path.join(base_dir, "1.labels_cleaned.csv")
input_img_dir = os.path.join(base_dir, "images")
output_img_dir = os.path.join(base_dir, "3.augmented_images")
output_csv_path = os.path.join(base_dir, "3.augmented_labels.csv")

num_augmentations_per_image = 10
save_every = 10
FINAL_SIZE = 1200
IMAGE_SHIFT_RIGHT = -30

# ==== Prepare folders ====
os.makedirs(output_img_dir, exist_ok=True)

# ==== Load CSV ====
df = pd.read_csv(input_csv)
augmented_rows = []

# ==== Augmentations ====
color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)

# ==== Helper functions ====
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

    # === Add original image as aug10 ===
    shifted = shift_image_right_crop(orig_img, IMAGE_SHIFT_RIGHT)
    masked = apply_circular_mask(shifted, diameter=FINAL_SIZE)
    cropped_img = crop_center(masked, FINAL_SIZE)

    aug_name = f"{os.path.splitext(row['New_Image_Name'])[0]}_aug{num_augmentations_per_image}.png"
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

    # === Augmented images (aug0 → aug9) ===
    for i in range(num_augmentations_per_image):
        angle = random.uniform(0, 360)

        jittered = color_jitter(orig_img)
        shifted = shift_image_right_crop(jittered, IMAGE_SHIFT_RIGHT)
        masked = apply_circular_mask(shifted, diameter=FINAL_SIZE)
        rotated = F.rotate(masked, angle, expand=True)
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
