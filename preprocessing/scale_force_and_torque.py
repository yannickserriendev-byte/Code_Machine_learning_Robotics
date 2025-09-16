"""
Preprocessing utility to scale force and torque values in the dataset.

- Scales Fx, Fy, Fz, Mx, My, Mz columns using provided calibration factors.
- Recalculates Ft (tangential force) from scaled Fx and Fy.
- Saves scaled CSV for downstream ML and analysis.
- Adjust input_csv_path for your dataset location.
"""

import pandas as pd
import numpy as np
import os

# ==== Configuration (update path as needed) ====
input_csv_path = "/scratch/yserrrien/data aqcuisition/1/full_dataset/3.augmented_labels.csv"
output_csv_path = input_csv_path.replace(".csv", "_scaled.csv")

# ==== Scale Vector ====
scale_vector = np.array([
    5.1694179576476,    # Fx
    5.1694179576476,    # Fy
    1.77487848937509,   # Fz
    221.35974008553,    # Mx (Tx)
    221.35974008553,    # My (Ty)
    211.642988806056    # Mz (Tz)
])

# ==== Load and process ====
df = pd.read_csv(input_csv_path)

# Ensure numeric and clean
force_cols = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
df[force_cols] = df[force_cols].apply(pd.to_numeric, errors='coerce').abs()

# Apply scaling
df[force_cols] = df[force_cols].div(scale_vector, axis=1)

# Recalculate Ft from scaled Fx and Fy
df['Ft'] = np.sqrt(df['Fx']**2 + df['Fy']**2)

# ==== Save result ====
df.to_csv(output_csv_path, index=False)
print(f"✅ Scaled data with recalculated Ft saved to:\n{output_csv_path}")
