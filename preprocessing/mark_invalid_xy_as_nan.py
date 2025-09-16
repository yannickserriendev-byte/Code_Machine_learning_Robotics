"""
Preprocessing utility to mark invalid XY positions and Indentor_Shape in the dataset.

- Sets Indentor_Shape to 'None' for weak force trials (Ft < 0.1).
- Sets X_Position_mm and Y_Position_mm to NaN for 'None' shape.
- Saves filtered CSV for downstream ML and analysis.
- Adjust input_csv path for your dataset location.
"""

import pandas as pd
import numpy as np
import os

# ==== Input CSV (update path as needed) ====
input_csv = "/scratch/yserrrien/data aqcuisition/1/full_dataset/3.augmented_labels_scaled.csv"
output_csv = input_csv.replace(".csv", "_filtered_indentor_and_xy_values_to_None_and_Nan.csv")

# ==== Load ====
df = pd.read_csv(input_csv)

# ==== Mark 'None' for weak force trials ====
df.loc[df['Ft'] < 0.1, 'Indentor_Shape'] = 'None'

# ==== Set (x, y) = NaN for 'None' shape ====
df.loc[df['Indentor_Shape'] == 'None', ['X_Position_mm', 'Y_Position_mm']] = np.nan

# ==== Save updated dataset ====
df.to_csv(output_csv, index=False)
print(f"✅ Filtered CSV with 'None' shapes and NaN (x,y) saved to:\n{output_csv}")
