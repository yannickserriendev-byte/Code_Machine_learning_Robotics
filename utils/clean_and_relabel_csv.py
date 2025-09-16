"""
Utility: Clean and relabel augmented labels CSV for tactile sensing pipeline.

- Converts force/moment columns to absolute values
- Sets Indentor_Shape to 'None' and (X, Y)_Position_mm to 'NaN' if Fz < 0.2
- Drops 'x' and 'y' columns if present
- Sorts by image and augmentation number
- Remaps Indentor_Shape to shape_class (None=0, others=1+)
- Usage: Set input/output paths, run script to clean and relabel CSV
"""
import pandas as pd
import numpy as np
import os
import re

def extract_image_aug_numbers(name):
    match = re.match(r"image_(\d+)_aug(\d+)\.png", name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return float('inf'), float('inf')

def apply_processing(input_path, output_path):
    df = pd.read_csv(input_path)
    force_columns = ['Fx', 'Fy', 'Fz', 'Ft', 'Mx', 'My', 'Mz']
    df[force_columns] = df[force_columns].abs()
    mask = df['Fz'] < 0.2
    df.loc[mask, 'Indentor_Shape'] = "None"
    df.loc[mask, ['X_Position_mm', 'Y_Position_mm']] = "NaN"
    df = df.drop(columns=[col for col in ['x', 'y'] if col in df.columns])
    sort_keys = df['New_Image_Name'].apply(extract_image_aug_numbers)
    df['Image_Number'] = [k[0] for k in sort_keys]
    df['Aug_Number'] = [k[1] for k in sort_keys]
    df = df.sort_values(by=['Image_Number', 'Aug_Number']) \
           .drop(columns=['Image_Number', 'Aug_Number']) \
           .reset_index(drop=True)
    unique_shapes = sorted(df['Indentor_Shape'].dropna().unique())
    if "None" in unique_shapes:
        unique_shapes.remove("None")
    shape_to_class = {"None": 0}
    for i, shape in enumerate(unique_shapes, start=1):
        shape_to_class[shape] = i
    df['shape_class'] = df['Indentor_Shape'].map(shape_to_class)
    df.to_csv(output_path, index=False)
    print(f"Saved updated labels to: {output_path}")

def main():
    base_folder = r"<SET TO YOUR DATASET FOLDER>"
    input_path = os.path.join(base_folder, "2.augmented_labels_full_pipeline_rotation_fix.csv")
    output_path = os.path.join(base_folder, "3.augmented_labels_full_pipeline_rotation_fix_ShapeWithNone_XYWithNan_based_on_Fz_indentorclassfix.csv")
    apply_processing(input_path, output_path)

if __name__ == "__main__":
    main()
