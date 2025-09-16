"""
Force-to-Image Mapping Module for Acquisition Pipeline

This module synchronizes force measurements with image frames for machine learning datasets.
The main challenge is that force sensors (1000+ Hz) and cameras (10-50 Hz) operate at 
different frequencies, so we need to match each image to its corresponding force measurement.

Key functions:
    - create_location_image_loss_file(): Diagnostic mapping to identify lost frames
    - create_final_mapping_file(): Final synchronized dataset for ML training

Example workflow:
    1. After acquisition, you have separate image and force data
    2. Use create_final_mapping_file() to create one CSV with each row containing:
       [Image_Number, Image_Timestamp, Force_Timestamp, Fx, Fy, Fz, Mx, My, Mz]
    3. This CSV becomes your ground-truth dataset for machine learning

File structure context:
    This assumes your data is in: Code_Machine_learning_Robotics/[your_trial_folder]/
    With subfolders: Images/ (contains image_1.tiff, image_2.tiff, ...)
                    Results/ (contains CSV files with force and timestamp data)
"""

import os
import numpy as np
import pandas as pd
import shutil

# -----------------------------------------------------------------------------
# Diagnostic mapping for frame loss analysis
# -----------------------------------------------------------------------------
def create_location_image_loss_file(mapping_filepath, acquisition_time, camera_frequency, camera_timestamps, 
                        force_timestamps, force_frequency, tolerance_factor=0.5):
    """
    Create a diagnostic CSV to analyze image loss and frame-force timing alignment.
    
    This function helps identify which frames were lost during acquisition and how
    well the image timestamps align with force measurements. Use this to debug
    acquisition timing issues before creating the final dataset.
    
    Parameters:
        mapping_filepath (str): Where to save diagnostic CSV (e.g., "trial_folder/frame_mapping.csv")
        acquisition_time (float): Total acquisition duration in seconds (e.g., 30.0)
        camera_frequency (int): Expected camera frame rate in fps (e.g., 10)
        camera_timestamps (list): Actual image timestamps from acquisition (in seconds)
        force_timestamps (list): Force measurement timestamps (in seconds)  
        force_frequency (int): Force sensor sampling rate in Hz (e.g., 1000)
        tolerance_factor (float): Timing tolerance for frame matching (default: 0.5)
    
    Creates CSV with columns:
        - Expected_Frame_Index: What frame number should be here (0, 1, 2, ...)
        - Expected_Frame_Timestamp: When this frame should have been captured
        - Acquired_Frame_Timestamp: When it was actually captured (empty if lost)
        - Acquired_Frame_Index: Actual index in the image list
        - Force_Sample_Index: Which force measurement matches this frame
        - Frame_Lost: 1 if frame was lost, 0 if captured successfully
    """
    # Calculate expected frame timing based on camera frequency
    expected_images = int(acquisition_time * camera_frequency)
    expected_interval = 1.0 / camera_frequency
    tolerance = expected_interval * tolerance_factor

    # Build diagnostic mapping for each expected frame
    mapping_list = []
    j = 0
    camera_timestamps_arr = np.array(camera_timestamps)
    force_timestamps_arr = np.array(force_timestamps)
    
    for i in range(expected_images):
        expected_time = i * expected_interval
        acquired_time = ""
        acquired_index = ""
        
        # Find if we have an image near the expected time
        while j < len(camera_timestamps_arr) and camera_timestamps_arr[j] < expected_time - tolerance:
            j += 1
        if j < len(camera_timestamps_arr) and abs(camera_timestamps_arr[j] - expected_time) <= tolerance:
            acquired_time = camera_timestamps_arr[j]
            acquired_index = j
            j += 1
        
        # Find corresponding force measurement
        force_index = ""
        force_time = ""
        if acquired_time != "":
            ideal_idx = int(round((acquired_time) * force_frequency))
            ideal_idx = max(0, min(ideal_idx, len(force_timestamps_arr) - 1))
            force_index = ideal_idx
            force_time = force_timestamps_arr[ideal_idx]
            
        mapping_list.append({
            "Expected_Frame_Index": i,
            "Expected_Frame_Timestamp": expected_time,
            "Acquired_Frame_Timestamp": acquired_time,
            "Acquired_Frame_Index": acquired_index,
            "Force_Sample_Index": force_index,
            "Force_Sample_Timestamp": force_time,
            "Frame_Lost": 1 if acquired_time == "" else 0
        })
        
    # Save diagnostic mapping to CSV
    df_mapping = pd.DataFrame(mapping_list)
    df_mapping.to_csv(mapping_filepath, index=False)
    print(f"Diagnostic mapping file saved to: {mapping_filepath}")
    return df_mapping

# -----------------------------------------------------------------------------
# Final mapping for machine learning dataset creation
# -----------------------------------------------------------------------------
def create_final_mapping_file(image_timestamps_path, timestamps_force_path, forces_csv_path, output_csv_path, image_folder):
    """
    Create the final synchronized dataset mapping each image to its force measurement.
    
    This is the main function for creating machine learning datasets. It takes all the
    separate acquisition files and creates one unified CSV where each row represents
    one training sample: [image info + synchronized force measurements].
    
    Important: This function also cleans up the image files by removing any images
    that were captured before force measurements started, and renumbers remaining
    images sequentially.
    
    Parameters:
        image_timestamps_path (str): Path to CSV with image timestamps
                                   Example: "trial_folder/image_timestamps.csv"
        timestamps_force_path (str): Path to CSV with force timestamps  
                                   Example: "trial_folder/force_timestamps.csv"
        forces_csv_path (str): Path to CSV with force measurements
                             Example: "trial_folder/force.csv"
        output_csv_path (str): Where to save final mapping CSV
                             Example: "trial_folder/Results/final_frame_force_mapping.csv"
        image_folder (str): Path to folder containing numbered image files
                          Example: "trial_folder/Images" (contains image_1.tiff, image_2.tiff, ...)
    
    Process:
        1. Load all timestamp and force data
        2. Remove images captured before force measurements began
        3. Renumber remaining images sequentially (image_1.tiff, image_2.tiff, ...)
        4. Match each image timestamp to closest force timestamp
        5. Create final CSV with columns: [Image_Number, Image_Timestamp, Force_Timestamp, Fx, Fy, Fz, Mx, My, Mz]
    
    Output CSV format:
        Each row represents one training sample for machine learning:
        - Image_Number: Sequential number (1, 2, 3, ...) corresponding to image_N.tiff
        - Image_Timestamp: When the image was captured (seconds)
        - Force_Timestamp: When the matched force was measured (seconds)  
        - Fx, Fy, Fz: Force components in Newtons
        - Mx, My, Mz: Moment components in Newton-millimeters
    """
    # Load all acquisition data files from the trial directory
    image_timestamps = pd.read_csv(image_timestamps_path)["Image_timestamps"].to_numpy()
    timestamps_force = pd.read_csv(timestamps_force_path)["Timestamp"].to_numpy()
    forces = pd.read_csv(forces_csv_path).to_numpy()

    if len(image_timestamps) == 0 or len(timestamps_force) == 0:
        raise ValueError("❌ One of the timestamp files is empty!")

    # Step 1: Remove images captured before force measurements started
    # This happens because camera and force acquisition may not start simultaneously
    valid_mask = image_timestamps >= timestamps_force[0]
    dropped_indices = np.where(~valid_mask)[0]
    kept_indices = np.where(valid_mask)[0]

    print(f"🗑️ Dropping {len(dropped_indices)} images that occurred before first force sample.")

    # Delete those early images from the Images folder
    for idx in dropped_indices:
        image_path = os.path.join(image_folder, f"image_{idx + 1}.tiff")
        if os.path.exists(image_path):
            os.remove(image_path)

    # Step 2: Renumber remaining images sequentially (image_1.tiff, image_2.tiff, ...)
    # This ensures the final dataset has consecutive image numbering starting from 1
    print(f"🔁 Renaming {len(kept_indices)} images to sequential indices...")

    for new_idx, original_idx in enumerate(kept_indices, start=1):
        old_path = os.path.join(image_folder, f"image_{original_idx + 1}.tiff")
        new_path = os.path.join(image_folder, f"image_{new_idx}.tiff")
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)

    # Step 3: Update image timestamps to match the kept images
    image_timestamps = image_timestamps[valid_mask]
    image_indices = np.arange(1, len(image_timestamps) + 1)

    # Step 4: For each image, find the closest force measurement in time
    matched_force_indices = []
    matched_force_timestamps = []
    for ts in image_timestamps:
        idx = np.argmin(np.abs(timestamps_force - ts))
        matched_force_indices.append(idx)
        matched_force_timestamps.append(timestamps_force[idx])

    # Step 5: Extract the corresponding force vectors (6-axis: Fx,Fy,Fz,Mx,My,Mz)
    def get_force_vec(idx):
        if 0 <= idx < len(forces):
            return forces[idx]
        return [None] * 6

    force_vectors = np.array([get_force_vec(idx) for idx in matched_force_indices])

    # Step 6: Create final dataset CSV with image-force pairs
    df_final = pd.DataFrame({
        "Image_Number": image_indices,
        "Image_Timestamp": image_timestamps,
        "Force_Timestamp": matched_force_timestamps,
        "Fx": force_vectors[:, 0],
        "Fy": force_vectors[:, 1],
        "Fz": force_vectors[:, 2],
        "Mx": force_vectors[:, 3],
        "My": force_vectors[:, 4],
        "Mz": force_vectors[:, 5],
    })

    # Save the final mapping - this becomes your ML training dataset
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_final.to_csv(output_csv_path, index=False)
    print(f"✅ Final mapping file saved to: {output_csv_path}")

    return df_final
