"""
Data Organization and Saving Module for Acquisition Results

This module handles the systematic organization and storage of all acquisition data
into a structured format ready for analysis and machine learning. After completing
your force and image acquisition, this module saves everything in an organized
folder structure with proper file naming and metadata.

Key function:
    - save_acquisition_results(): Master function to save all acquisition data

What gets saved:
    - Images as sequential TIFF files (image_1.tiff, image_2.tiff, ...)
    - Force measurements as CSV with 6-axis data [Fx,Fy,Fz,Mx,My,Mz]
    - Image timestamps synchronized with force timestamps
    - Acquisition metadata and parameters for reproducibility
    - Data info file with units and acquisition settings

Typical workflow:
    1. Complete your acquisition using _01_acquisition.py
    2. Call save_acquisition_results() with all collected data
    3. Your trial folder becomes a complete dataset ready for processing

Directory structure created:
    Code_Machine_learning_Robotics/[trial_folder]/
    ├── Images/
    │   ├── image_1.tiff
    │   ├── image_2.tiff
    │   └── ...
    ├── force.csv
    ├── image_timestamps.csv
    ├── force_timestamps.csv
    └── data_info.txt

This standardized structure ensures compatibility with the rest of the processing pipeline
and makes your data easily shareable and reproducible for scientific work.
"""

import os
import pandas as pd
import cv2
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Main data saving function
# -----------------------------------------------------------------------------
def save_acquisition_results(
    image_data,
    timestamps,
    timestamps_force,
    forces,
    force_acquisition_frequency,
    buffer_size,
    acquisition_frequency_camera,
    elapsed_time,
    trial_dir
):
    """
    Save all acquisition data (images, forces, timestamps) to an organized folder structure.
    
    This function creates a complete trial dataset by saving:
        - Images as numbered TIFF files (image_1.tiff, image_2.tiff, etc.)
        - Force measurements as CSV with columns [Fx, Fy, Fz, Mx, My, Mz]
        - Image timestamps synchronized with force timestamps
        - Metadata file with acquisition parameters
    
    Parameters:
        image_data (list): List of numpy arrays, each representing one captured image
        timestamps (list): List of floats, timestamp for each image (in seconds)
        timestamps_force (list): List of floats, timestamp for each force sample batch
        forces (numpy.array): 2D array with shape (n_samples, 6) containing [Fx,Fy,Fz,Mx,My,Mz]
        force_acquisition_frequency (int): Force sensor sampling rate in Hz (e.g., 1000)
        buffer_size (int): Number of force samples per callback (e.g., 100)
        acquisition_frequency_camera (int): Camera frame rate in fps (e.g., 10)
        elapsed_time (float): Total time the acquisition ran (in seconds)
        trial_dir (str): Full path to trial folder (e.g., "C:/Data/Trial_001")
    
    Creates folder structure:
        trial_dir/
        ├── Images/
        │   ├── image_1.tiff
        │   ├── image_2.tiff
        │   └── ...
        ├── force.csv
        ├── image_timestamps.csv
        ├── force_timestamps.csv
        └── data_info.txt
    """
    # Get the trial directory path (should already exist from setup)
    DIR = trial_dir
    print("Saving force data...")

    # Create CSV files for all force and timestamp data
    df_force = pd.DataFrame(forces, columns=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
    df_image_timestamps = pd.DataFrame(timestamps, columns=['Image_timestamps'])
    df_force_timestamps_callback = pd.DataFrame(timestamps_force, columns=['Force_timestamps_callback_func'])

    # Save force data and timestamps to CSV files
    df_force.to_csv(os.path.join(DIR, 'force.csv'), index=False)
    df_image_timestamps.to_csv(os.path.join(DIR, 'image_timestamps.csv'), index=False)
    df_force_timestamps_callback.to_csv(os.path.join(DIR, 'force_timestamps_callback_func.csv'), index=False)

    # Also save the absolute force timestamps (main file for analysis)
    df_force_timestamps_absolute = pd.DataFrame(timestamps_force, columns=['Force_Timestamps'])
    df_force_timestamps_absolute.to_csv(os.path.join(DIR, 'force_timestamps.csv'), index=False)

    # Create 'Images' subfolder and save all captured images
    images_dir = os.path.join(DIR, "Images")
    os.makedirs(images_dir, exist_ok=True)
    print("Saving images to disk in folder 'Images'...")

    # Save each image as a numbered TIFF file (image_1.tiff, image_2.tiff, etc.)
    for idx, img in tqdm(enumerate(image_data), total=len(image_data), desc="Saving images"):
        image_filename = os.path.join(images_dir, f"image_{idx + 1}.tiff")
        cv2.imwrite(image_filename, img)

    # Create metadata file with acquisition parameters for future reference
    # Create metadata file with acquisition parameters for future reference
    with open(os.path.join(DIR, 'data_info.txt'), 'w') as file:
        file.write(f"Force acquisition frequency: {force_acquisition_frequency} Hz\n")
        file.write(f"Force buffer size: {buffer_size}\n")
        file.write(f"Image acquisition frequency: {acquisition_frequency_camera} Hz\n")
        file.write(f"Elapsed time acquisition: {elapsed_time:.4f} s\n")
        file.write("Force timestamps (absolute): force_timestamps.csv\n")
        file.write("Unit measure: ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']\n")

    print('[OK] All acquisition data saved successfully to trial folder.')
