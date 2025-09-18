"""
Post-processing script for tactile sensing data augmentation labels.

This script performs the following operations on a CSV file containing augmented labels:
1. Converts all force and moment columns (Fx, Fy, Fz, Ft, Mx, My, Mz) to absolute values.
2. For rows where Fz < 0.2:
   - Sets 'Indentor_Shape' to "None"
   - Sets 'X_Position_mm' and 'Y_Position_mm' to NaN
3. Removes columns 'x' and 'y' if present (from augmentation step)
4. Sorts the DataFrame by image and augmentation number extracted from 'New_Image_Name'
5. Remaps 'Indentor_Shape' to 'shape_class':
   - "None" → 0
   - Other shapes → increasing integers starting from 1
6. Saves the cleaned DataFrame to a new CSV file.

Usage:
------
Edit the input_path and output_path variables in main() to match your filenames.
Run: python postprocess_labels.py
"""


import os
import pandas as pd
import numpy as np
import re
from datetime import datetime

def extract_image_aug_numbers(name):
    """Extract image and augmentation numbers from filename."""
    match = re.match(r"image_(\d+)_aug(\d+)\.png", name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return float('inf'), float('inf')  # fallback for malformed names

def apply_processing(input_path, output_path):
    df = pd.read_csv(input_path)

    # Step 1: Convert force/moment columns to absolute values
    force_columns = ['Fx', 'Fy', 'Fz', 'Ft', 'Mx', 'My', 'Mz']
    df[force_columns] = df[force_columns].abs()

    # Step 2: Set Indentor_Shape to None and original X/Y to NaN for Fz < 0.2
    mask = df['Fz'] < 0.2
    df.loc[mask, 'Indentor_Shape'] = "None"
    df.loc[mask, ['X_Position_mm_after_rotation', 'Y_Position_mm_after_rotation']] = np.nan

    # Step 3: Rename rotated position columns if present
    if 'x' in df.columns:
        df = df.rename(columns={'x': 'Contact_X_mm_after_rotation'})
    if 'y' in df.columns:
        df = df.rename(columns={'y': 'Contact_Y_mm_after_rotation'})

    # Step 4: Sort by image and augmentation number
    sort_keys = df['New_Image_Name'].apply(extract_image_aug_numbers)
    df['Image_Number'] = [k[0] for k in sort_keys]
    df['Aug_Number'] = [k[1] for k in sort_keys]
    df = df.sort_values(by=['Image_Number', 'Aug_Number'])
    df = df.drop(columns=['Image_Number', 'Aug_Number'])
    df = df.reset_index(drop=True)

    # Step 5: Remap Indentor_Shape to shape_class
    unique_shapes = sorted(df['Indentor_Shape'].dropna().unique())
    if "None" in unique_shapes:
        unique_shapes.remove("None")
    shape_to_class = {"None": 0}
    for i, shape in enumerate(unique_shapes, start=1):
        shape_to_class[shape] = i
    df['shape_class'] = df['Indentor_Shape'].map(shape_to_class)

    # Print shape class mapping for user verification
    print("Shape class mapping (Indentor_Shape → shape_class):")
    for shape, cls in shape_to_class.items():
        print(f"  {shape}: {cls}")
    print(f"Total different shape classes found (excluding 'None'): {len(shape_to_class)-1}")

    # Save result
    df.to_csv(output_path, index=False)
    print(f"Saved updated labels to: {output_path}")

def main():
    # Set up base directory and paths (same principle as augmentation)
    # User provides the input file path directly
    base_dir = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\code\code full pipeline\All code\Code from laptop\Testing_data_del\Data\full_dataset"  # <-- EDIT THIS LINE
    input_path =os.path.join(base_dir, "1.aug_lab_s_0918_1200.csv")
    # Detect environment (desktop or supercomputer) from input filename
    input_filename = os.path.basename(input_path)
    env_match = re.search(r"1\.aug_lab_(d|s)_(\d{4}_\d{4})\.csv", input_filename)
    if not env_match:
        raise ValueError("Input filename does not match expected pattern for environment and timestamp.")
    env_suffix = env_match.group(1)  # 's' for desktop, 'sc' for supercomputer
    timestamp = env_match.group(2)
    base_dir = os.path.dirname(input_path)
    output_path = os.path.join(base_dir, f"2.aug_lab_postproc_{env_suffix}_{timestamp}.csv")
    apply_processing(input_path, output_path)

if __name__ == "__main__":
    main()
