"""
Preprocessing utility to correct shape class mapping and set invalid (x, y) to NaN.

- Sets (x, y) to NaN for rows with Indentor_Shape == 'None'.
- Maps all unique Indentor_Shape values to integer class labels (including 'None').
- Adds 'shape_class' column to the dataset for classification tasks.
- Saves updated CSV and prints mapping for user reference.
- Adjust input_csv path for your dataset location.
"""

import pandas as pd
import numpy as np
import os

# ==== Input/Output Paths (update as needed) ====
input_csv = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\2. Data creation\data aqcuisition\1\full_dataset\3.augmented_labels_scaled_filtered_indentor_and_xy_values_to_None_and_Nan.csv"
output_csv = input_csv.replace(".csv", "_Indenter_Shape_correction.csv")

# ==== Load CSV — treat 'None' as string ====
df = pd.read_csv(input_csv, keep_default_na=False, na_values=[])

# ==== Set (x, y) = NaN where shape is 'None' (as string) ====
df.loc[df['Indentor_Shape'] == 'None', ['x', 'y']] = np.nan

# ==== Map all unique shape names (incl. 'None') ====
unique_shapes = sorted(df['Indentor_Shape'].unique())  # no NaN present now
shape_to_class = {shape: idx for idx, shape in enumerate(unique_shapes)}
df['shape_class'] = df['Indentor_Shape'].map(shape_to_class)

# ==== Save result ====
df.to_csv(output_csv, index=False)
print(f"✅ Updated CSV saved with shape_class mapping:\n{shape_to_class}")
print(f"📄 Output: {output_csv}")
