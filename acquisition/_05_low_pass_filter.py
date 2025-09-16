"""
Low-Pass Filtering Module for Force Signal Processing

This module provides tools to smooth noisy force and torque signals using Butterworth filters.
Force sensors often produce noisy signals that need filtering before machine learning or analysis.

Key functions:
    - butter_lowpass_filter(): Apply Butterworth filter to any signal
    - filter_final_frame_force_mapping(): Apply filtering to your final ML dataset CSV

Why filter force signals?
    Raw force sensors pick up vibrations, electrical noise, and mechanical jitter.
    Filtering removes high-frequency noise while preserving the actual force trends
    that are important for tactile sensing and machine learning.

Typical workflow:
    1. After creating your final image-force mapping CSV
    2. Run filter_final_frame_force_mapping() to create a filtered version
    3. Compare filtered vs unfiltered data to choose the best version for your analysis

Directory context:
    Input: Code_Machine_learning_Robotics/[trial_folder]/Results/unfiltered_final_frame_force_mapping.csv
    Output: Code_Machine_learning_Robotics/[trial_folder]/Results/filtered_final_frame_force_mapping.csv
"""

import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# -----------------------------------------------------------------------------
# Core filtering function
# -----------------------------------------------------------------------------
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter to smooth noisy signals.
    
    This removes high-frequency noise while preserving the underlying signal trends.
    Commonly used for force sensor data to remove vibrations and electrical noise.
    
    Parameters:
        data (array): Signal to filter (e.g., force measurements over time)
        cutoff (float): Cutoff frequency in Hz (frequencies above this are removed)
                       Example: 5 Hz removes vibrations above 5 Hz
        fs (float): Sampling frequency of the signal in Hz
                   Example: 10 Hz for camera-rate force data
        order (int): Filter order (higher = steeper cutoff, default=4)
    
    Returns:
        array: Filtered signal with same length as input
    
    Example:
        # Filter 10 Hz force data, removing frequencies above 2 Hz
        filtered_force = butter_lowpass_filter(force_data, cutoff=2, fs=10, order=4)
    """
    # Calculate normalized cutoff frequency for Butterworth filter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

# -----------------------------------------------------------------------------
# Filter final mapping CSV for machine learning
# -----------------------------------------------------------------------------
def filter_final_frame_force_mapping(mapping_path, output_folder, cutoff_ratio=0.1, order=4):
    """
    Apply low-pass filtering to your final image-force mapping CSV file.
    
    This function takes the CSV created by create_final_mapping_file() and applies
    filtering to all force/torque columns, creating a cleaned version for machine learning.
    Also computes tangential force (Ft) as the magnitude of horizontal forces.
    
    Parameters:
        mapping_path (str): Path to your unfiltered mapping CSV
                          Example: "Code_Machine_learning_Robotics/Trial_001/Results/unfiltered_final_frame_force_mapping.csv"
        output_folder (str): Where to save the filtered CSV  
                           Example: "Code_Machine_learning_Robotics/Trial_001/Results/"
        cutoff_ratio (float): Fraction of Nyquist frequency to use as cutoff (default: 0.1)
                            Lower values = more aggressive filtering
                            Example: 0.1 means cutoff at 10% of max detectable frequency
        order (int): Filter order for steepness (default: 4)
    
    Process:
        1. Load the unfiltered mapping CSV with columns [Image_Number, Image_Timestamp, Force_Timestamp, Fx, Fy, Fz, Mx, My, Mz]
        2. Calculate effective sampling frequency from image timestamps
        3. Apply Butterworth filter to all force/torque columns  
        4. Compute tangential force: Ft = sqrt(Fx² + Fy²)
        5. Save filtered version as "filtered_final_frame_force_mapping.csv"
    
    Output:
        Creates filtered CSV with same structure plus additional Ft column.
        Use this for machine learning if your raw signals are too noisy.
    """
    # Load the unfiltered mapping CSV
    df = pd.read_csv(mapping_path)

    # Verify timestamp column exists (handle possible naming variations)
    timestamp_col = "Image_Timestamp"
    if timestamp_col not in df.columns:
        raise KeyError(f"Column '{timestamp_col}' not found. Available columns: {df.columns.tolist()}")

    timestamps = df[timestamp_col].values
    if len(timestamps) < 2:
        raise ValueError("Not enough timestamp values to compute sampling rate.")

    # Calculate effective sampling frequency from timestamp intervals
    dt = np.diff(timestamps)
    fs = 1 / np.mean(dt)
    nyq = 0.5 * fs
    cutoff = nyq * cutoff_ratio

    print(f"📈 Effective sampling frequency estimated: {fs:.2f} Hz")
    print(f"🎚 Cutoff frequency applied: {cutoff:.2f} Hz (Order: {order})")

    # Apply filtering to all force and torque columns
    force_cols = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    df_filtered = df.copy()

    for col in force_cols:
        if col in df.columns:
            try:
                df_filtered[col] = butter_lowpass_filter(df[col].values, cutoff, fs, order)
            except Exception as e:
                print(f"⚠️ Could not filter column '{col}': {e}")
        else:
            print(f"⚠️ Column '{col}' not found in mapping file.")

    # Compute tangential force magnitude (useful for tactile sensing analysis)
    if 'Fx' in df_filtered.columns and 'Fy' in df_filtered.columns:
        df_filtered['Ft'] = np.sqrt(df_filtered['Fx'] ** 2 + df_filtered['Fy'] ** 2)
        print("✅ Tangential force Ft computed as sqrt(Fx² + Fy²).")

    # Save the filtered mapping CSV
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "filtered_final_frame_force_mapping.csv")
    df_filtered.to_csv(output_path, index=False)
    print(f"✅ Filtered mapping saved to: {output_path}")
