"""
Utility: Combine raw image and label data from multiple trials into a single full dataset for tactile sensing pipeline.

- Merges position and force CSVs for each trial
- Copies images to output folder with new sequential names
- Creates a unified labels CSV for all images
- Usage: Set base_dir and output_dir, run script to generate full dataset
"""
import os
import pandas as pd
import shutil
from tqdm import tqdm
import time

def create_full_dataset(base_dir, output_dir, start_idx=1, end_idx=20):
    os.makedirs(os.path.join(output_dir, "0.images"), exist_ok=True)
    all_labels = []
    global_image_counter = 1
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
        df_pos = pd.read_csv(pos_file)
        df_force = pd.read_csv(force_file)
        merged = pd.merge(df_force, df_pos, on="Image_Number", how="inner")
        for _, row in merged.iterrows():
            old_image_name = f"image_{int(row['Image_Number'])}.tiff"
            old_image_path = os.path.join(images_path, old_image_name)
            new_image_name = f"image_{global_image_counter}.tiff"
            new_image_path = os.path.join(output_dir, "0.images", new_image_name)
            if os.path.exists(old_image_path):
                shutil.copy(old_image_path, new_image_path)
                label = {
                    "New_Image_Name": new_image_name,
                    "Fx": row.get("Fx", None),
                    "Fy": row.get("Fy", None),
                    "Fz": row.get("Fz", None),
                    "Mx": row.get("Mx", None),
                    "My": row.get("My", None),
                    "Mz": row.get("Mz", None),
                    "Ft": row.get("Ft", None),
                    "X_Position_mm": row.get("X_Position_mm", None),
                    "Y_Position_mm": row.get("Y_Position_mm", None),
                    "Indentor_Shape": row.get("Indentor_Shape", None)
                }
                all_labels.append(label)
                global_image_counter += 1
            else:
                print(f"⚠️ Missing image file: {old_image_path}")
        elapsed = time.time() - start_time
        folders_remaining = (end_idx - i)
        est_remaining_time = folders_remaining * elapsed
        print(f"✅ Done {i - start_idx + 1}/{total_trials} | Estimated time left: {est_remaining_time:.1f} seconds")
    df_labels = pd.DataFrame(all_labels)
    labels_csv_path = os.path.join(output_dir, "0.labels.csv")
    df_labels.to_csv(labels_csv_path, index=False)
    print(f"\n✅ Dataset creation complete. {len(df_labels)} images labeled.")
    print(f"📁 Output written to: {labels_csv_path}")

if __name__ == "__main__":
    base_directory = r"<SET TO YOUR RAW DATA FOLDER>"
    output_dataset_dir = os.path.join(base_directory, "full_dataset")
    create_full_dataset(base_directory, output_dataset_dir)
