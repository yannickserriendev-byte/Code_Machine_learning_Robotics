"""
Preprocessing utility to correct XY contact points using rotation angle.

- Rotates (X_Position_mm, Y_Position_mm) by 'rotation_angle' for each row.
- Adds corrected 'x' and 'y' columns to the dataset.
- Saves corrected CSV for downstream ML and analysis.
- Adjust base_dir and csv_path for your dataset location.
"""

import os
import math
import pandas as pd

# === Paths (update as needed) ===
base_dir = "/scratch/yserrrien/data aqcuisition/1/full_dataset"
csv_path = os.path.join(base_dir, "3.augmented_labels_scaled_filtered_indentor_and_xy_values_to_None_and_Nan.csv")
corrected_csv_path = os.path.join(base_dir, "3.augmented_labels_scaled_filtered_indentor_and_xy_values_to_None_and_Nan_y_value_corrected.csv")

# === Load CSV ===
df = pd.read_csv(csv_path)

# === Rotation Helper ===
def rotate_point(x, y, angle_deg):
    angle_rad = math.radians(angle_deg)
    x_new = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    y_new = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    return x_new, y_new

# === Fix Contact Points ===
corrected_x = []
corrected_y = []

for _, row in df.iterrows():
    x, y = row["X_Position_mm"], row["Y_Position_mm"]
    angle = row["rotation_angle"]

    if angle == 0.0:
        corrected_x.append(x)
        corrected_y.append(y)
    else:
        x_rot, y_rot = rotate_point(x, y, angle)
        corrected_x.append(x_rot)
        corrected_y.append(y_rot)

df["x"] = corrected_x
df["y"] = corrected_y

# === Save Corrected File ===
df.to_csv(corrected_csv_path, index=False)
print(f"✅ Corrected contact points saved to: {corrected_csv_path}")
