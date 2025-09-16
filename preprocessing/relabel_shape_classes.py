"""
Preprocessing utility to encode shape classes for ML.

- Maps Indentor_Shape to integer class labels (None = 0, others = 1, 2, ...).
- Adds 'shape_class' column to the dataset for classification tasks.
- Saves updated CSV and prints mapping for user reference.
- Adjust input_csv path for your dataset location.
"""

import pandas as pd
import os

# ==== Load CSV (update path as needed) ====
input_csv = "/scratch/yserrrien/data aqcuisition/1/full_dataset/3.augmented_labels_scaled_filtered_indentor_and_xy_values_to_None_and_Nan.csv"
output_csv = input_csv.replace(".csv", "_with_shape_class.csv")

df = pd.read_csv(input_csv)

# ==== Encode shape_class with 'None' as class 0 ====
unique_shapes = sorted(df['Indentor_Shape'].dropna().unique())
shape_mapping = {}

# Ensure 'None' is first
if 'None' in unique_shapes:
    shape_mapping['None'] = 0
    remaining_shapes = [s for s in unique_shapes if s != 'None']
else:
    remaining_shapes = unique_shapes

# Assign classes 1, 2, ... to other shapes
for i, shape in enumerate(remaining_shapes, start=1):
    shape_mapping[shape] = i

# Apply mapping
df['shape_class'] = df['Indentor_Shape'].map(shape_mapping)

# ==== Save ====
df.to_csv(output_csv, index=False)
print(f"✅ Saved updated CSV with shape_class to:\n{output_csv}")
print("🔢 Shape mapping used:")
for k, v in shape_mapping.items():
    print(f"  {k:10s} → {v}")
