"""
Tactile Sensor Data Combination Script

This script combines multiple trial folders from tactile sensor data collection into a unified dataset.
It processes both photoelastic fringe pattern images and their associated sensor measurements.

Required Directory Structure:
    base_dir/
    ├── Sensor_1_clear_image_03_06_2025_1/    # Trial folder format: SensorID_ImageType_Date_TrialNumber
    │   ├── Images/                            # Contains raw .tiff images from photoelastic sensor
    │   └── Results/                           # Contains measurement data
    │       ├── positions_indentation.csv      # X,Y coordinates of indentation points
    │       └── unfiltered_final_frame_force_mapping.csv  # Force/torque sensor readings
    └── ... (more trial folders)

Output Structure:
    output_dir/full_dataset/
    ├── 0.images/                # Combined and renumbered images
    │   ├── image_1.tiff        # Images renamed with sequential numbering
    │   └── ...
    └── 0.labels.csv            # Combined metadata file with:
                               # - Force components (Fx, Fy, Fz)
                               # - Torque components (Mx, My, Mz)
                               # - Total force (Ft)
                               # - Position data (X_Position_mm, Y_Position_mm)
                               # - Indentation shape information

Important Notes:
1. Trial Folder Naming:
   - Default format: "Sensor_1_clear_image_MM_DD_YYYY_N"
   - To use different naming: Modify trial_name pattern in create_full_dataset()

2. Data Requirements:
   - Images must be in .tiff format
   - CSV files must contain specific column names (Image_Number, force components, positions)
   - Position and force data must share Image_Number for correct matching

3. Performance:
   - Uses pandas for efficient data merging
   - Includes progress tracking with time estimates
   - Handles missing data gracefully
"""

import os
import pandas as pd
import shutil
from tqdm import tqdm
import time

def create_full_dataset(base_dir, output_dir, start_idx=1, end_idx=20):
    """
    Creates a unified dataset by combining multiple trial folders into a single structured dataset.
    
    Args:
        base_dir (str): Root directory containing all trial folders
        output_dir (str): Directory where the combined dataset will be saved
        start_idx (int): First trial number to process (default: 1)
        end_idx (int): Last trial number to process (default: 20)
    
    The function performs the following steps:
    1. Creates output directory structure
    2. For each trial folder:
       - Reads position data (X,Y coordinates)
       - Reads force/torque measurements
       - Merges data based on Image_Number
       - Copies and renames images with global counter
    3. Combines all metadata into a single CSV file
    
    Error Handling:
    - Skips missing CSV files with warning
    - Skips missing image files with warning
    - Uses .get() for safe column access
    """
    os.makedirs(os.path.join(output_dir, "0.images"), exist_ok=True)
    all_labels = []
    global_image_counter = 1  # Used for sequential renaming of images

    total_trials = end_idx - start_idx + 1
    print(f"🗂 Processing {total_trials} folders...\n")

    for i in tqdm(range(start_idx, end_idx + 1), desc="Processing Trials", unit="folder"):
        trial_name = f"Sensor_1_clear_image_03_06_2025_{i}"
        trial_path = os.path.join(base_dir, trial_name)

        start_time = time.time()

        images_path = os.path.join(trial_path, "Images")
        results_path = os.path.join(trial_path, "Results")

        pos_file = os.path.join(results_path, "positions_indentation.csv")
        force_file = os.path.join(results_path, "unfiltered_final_frame_force_mapping.csv")

        if not os.path.exists(pos_file) or not os.path.exists(force_file):
            print(f"⚠️ Skipping {trial_name}: Missing CSV file(s).")
            continue

        # Load and merge measurement data
        df_pos = pd.read_csv(pos_file)     # Contains X,Y coordinates of indentation points
        df_force = pd.read_csv(force_file)  # Contains force/torque measurements
        
        # Inner join ensures we only process images that have both position and force data
        # This prevents incomplete or inconsistent data in the final dataset
        merged = pd.merge(df_force, df_pos, on="Image_Number", how="inner")

        for _, row in merged.iterrows():
            old_image_name = f"image_{int(row['Image_Number'])}.tiff"
            old_image_path = os.path.join(images_path, old_image_name)

            new_image_name = f"image_{global_image_counter}.tiff"
            new_image_path = os.path.join(output_dir, "0.images", new_image_name)

            if os.path.exists(old_image_path):
                shutil.copy(old_image_path, new_image_path)
                # Create label dictionary with all metadata
                # Using .get() with None default for robustness against missing columns
                label = {
                    "New_Image_Name": new_image_name,
                    # Force components (N)
                    "Fx": row.get("Fx", None),      # Force in X direction
                    "Fy": row.get("Fy", None),      # Force in Y direction
                    "Fz": row.get("Fz", None),      # Normal force (Z direction)
                    # Torque components (N⋅mm)
                    "Mx": row.get("Mx", None),      # Moment around X axis
                    "My": row.get("My", None),      # Moment around Y axis
                    "Mz": row.get("Mz", None),      # Moment around Z axis
                    "Ft": row.get("Ft", None),      # Total force magnitude
                    # Position data (mm)
                    "X_Position_mm": row.get("X_Position_mm", None),    # X coordinate in mm
                    "Y_Position_mm": row.get("Y_Position_mm", None),    # Y coordinate in mm
                    # Experimental parameters
                    "Indentor_Shape": row.get("Indentor_Shape", None)  # Shape of the indentation tool
                }
                all_labels.append(label)
                global_image_counter += 1
            else:
                print(f"⚠️ Missing image file: {old_image_path}")

        elapsed = time.time() - start_time
        folders_remaining = (end_idx - i)
        est_remaining_time = folders_remaining * elapsed
        print(f"✅ Done {i - start_idx + 1}/{total_trials} | Estimated time left: {est_remaining_time:.1f} seconds")

    # Save final label file
    df_labels = pd.DataFrame(all_labels)
    labels_csv_path = os.path.join(output_dir, "0.labels.csv")
    df_labels.to_csv(labels_csv_path, index=False)
    print(f"\n✅ Dataset creation complete. {len(df_labels)} images labeled.")
    print(f"📁 Output written to: {labels_csv_path}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Base directory containing the trial folders
    base_directory = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\code\code full pipeline\All code\Code from laptop\Testing_data_del\Data"
    
    # Output directory for the combined dataset
    # Will create: full_dataset/0.images/ and full_dataset/0.labels.csv
    output_dataset_dir = os.path.join(base_directory, "full_dataset")
    
    # Process all trial folders and combine data
    create_full_dataset(base_directory, output_dataset_dir)
