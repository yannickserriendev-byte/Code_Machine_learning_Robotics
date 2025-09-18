
"""
Expert-level post-processing script for tactile sensing data augmentation labels.

This script cleans and standardizes augmented label CSVs for downstream machine learning tasks.
Key operations:
1. Converts all force and moment columns (Fx, Fy, Fz, Ft, Mx, My, Mz) to absolute values for physical consistency.
2. For rows where Fz < 0.2 (no contact):
    - Sets 'Indentor_Shape' to "None"
    - Sets rotated X/Y position columns to NaN (invalid position)
3. Renames position columns from augmentation if present ('x', 'y' → 'Contact_X_mm_after_rotation', 'Contact_Y_mm_after_rotation').
4. Sorts the DataFrame by image and augmentation number for reproducible ordering.
5. Remaps 'Indentor_Shape' to integer 'shape_class' for model compatibility:
    - "None" → 0
    - Other shapes → increasing integers starting from 1
6. Prints shape class mapping for user verification.
7. Saves the cleaned DataFrame to a new CSV file.

Usage:
------
Edit the input_path variable in main() to match your filenames.
Run: python postprocess_labels.py
"""


import os
import pandas as pd
import numpy as np
import re
from datetime import datetime

def extract_image_aug_numbers(name):
    """
    Extract image and augmentation numbers from filename for sorting.
    Returns (image_number, aug_number) as integers, or (inf, inf) if malformed.
    """
    match = re.match(r"image_(\d+)_aug(\d+)\.png", name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return float('inf'), float('inf')  # fallback for malformed names

def apply_processing(input_path, output_path):
    # Load CSV into DataFrame
    df = pd.read_csv(input_path)

    # --- Step 1: Convert force/moment columns to absolute values ---
    # Ensures all physical quantities are positive for ML consistency
    force_columns = ['Fx', 'Fy', 'Fz', 'Ft', 'Mx', 'My', 'Mz']
    df[force_columns] = df[force_columns].abs()

    # --- Step 2: Set Indentor_Shape to None and rotated X/Y to NaN for Fz < 0.2 ---
    # Rows with Fz < 0.2 are considered 'no contact' and should not have valid shape or position
    mask = df['Fz'] < 0.2
    df.loc[mask, 'Indentor_Shape'] = "None"
    df.loc[mask, ['X_Position_mm_after_rotation', 'Y_Position_mm_after_rotation']] = np.nan

    # --- Step 3: Rename rotated position columns if present ---
    # Handles legacy column names from augmentation step
    if 'x' in df.columns:
        df = df.rename(columns={'x': 'Contact_X_mm_after_rotation'})
    if 'y' in df.columns:
        df = df.rename(columns={'y': 'Contact_Y_mm_after_rotation'})

    # --- Step 4: Sort by image and augmentation number ---
    # Sorting ensures reproducibility and correct ordering for downstream tasks
    sort_keys = df['New_Image_Name'].apply(extract_image_aug_numbers)
    df['Image_Number'] = [k[0] for k in sort_keys]
    df['Aug_Number'] = [k[1] for k in sort_keys]
    df = df.sort_values(by=['Image_Number', 'Aug_Number'])
    df = df.drop(columns=['Image_Number', 'Aug_Number'])
    df = df.reset_index(drop=True)

    # --- Step 5: Remap Indentor_Shape to shape_class ---
    # Maps each unique shape to an integer class for ML compatibility
    unique_shapes = sorted(df['Indentor_Shape'].dropna().unique())
    if "None" in unique_shapes:
        unique_shapes.remove("None")
    shape_to_class = {"None": 0}  # 'None' always maps to class 0
    for i, shape in enumerate(unique_shapes, start=1):
        shape_to_class[shape] = i
    df['shape_class'] = df['Indentor_Shape'].map(shape_to_class)

    # --- Print shape class mapping for user verification ---
    # Helps user double-check shape assignments and class counts
    print("Shape class mapping (Indentor_Shape → shape_class):")
    for shape, cls in shape_to_class.items():
        print(f"  {shape}: {cls}")
    print(f"Total different shape classes found (excluding 'None'): {len(shape_to_class)-1}")

    # --- Save result ---
    # Writes cleaned DataFrame to output CSV
    df.to_csv(output_path, index=False)
    print(f"Saved updated labels to: {output_path}")

def main():
    # --- Path setup ---
    # User provides the input file path directly below
    base_dir = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\code\code full pipeline\All code\Code from laptop\Testing_data_del\Data\full_dataset"  # <-- EDIT THIS LINE
    input_path = os.path.join(base_dir, "1.aug_lab_s_0918_1200.csv")

    # Detect environment (desktop or supercomputer) from input filename for output naming
    input_filename = os.path.basename(input_path)
    env_match = re.search(r"1\.aug_lab_(d|s)_(\d{4}_\d{4})\.csv", input_filename)
    if not env_match:
        raise ValueError("Input filename does not match expected pattern for environment and timestamp.")
    env_suffix = env_match.group(1)  # 's' for desktop, 'd' for supercomputer
    timestamp = env_match.group(2)
    base_dir = os.path.dirname(input_path)
    output_path = os.path.join(base_dir, f"2.aug_lab_postproc_{env_suffix}_{timestamp}.csv")

    # Run postprocessing
    apply_processing(input_path, output_path)

if __name__ == "__main__":
    main()
