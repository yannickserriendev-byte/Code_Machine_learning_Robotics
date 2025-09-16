"""
Plotting and Visualization Module for Acquisition Results

This module creates publication-quality plots of force and torque data over time.
After creating your final image-force mapping, use these tools to visualize:
    - All 6-axis force/torque channels (Fx, Fy, Fz, Mx, My, Mz)
    - Normal force (Fz) and tangential force trends
    - Dual x-axes showing both time (seconds) and frame numbers

Key features:
    - Highlights the most important force components (Fz normal, tangential)
    - Shows all channels in light gray for context
    - Dual x-axis for easy correlation with image frame numbers

Typical usage:
    1. After creating filtered or unfiltered mapping CSV
    2. Call plot_final_mapping_forces() to visualize force trends
    3. Use plots to verify data quality and identify interesting force patterns

Directory context:
    Input: Code_Machine_learning_Robotics/[trial_folder]/Results/[filtered_or_unfiltered]_final_frame_force_mapping.csv
    Output: Saves plots or displays them interactively
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------------------------
# Main plotting function for force visualization
# -----------------------------------------------------------------------------
def plot_final_mapping_forces(final_mapping_path, save_path=None):
    """
    Create a comprehensive visualization of force/torque data over time and frame index.
    
    This function generates a dual-axis plot showing:
        - Bottom x-axis: Time in seconds (from image timestamps)
        - Top x-axis: Frame numbers (corresponding to image_N.tiff files)
        - Y-axis: Force values in Newtons and Torque in Newton-millimeters
        - Highlighted: Normal force (Fz) and tangential force components
        - Background: All 6 channels in light gray for context
    
    Parameters:
        final_mapping_path (str): Path to your final mapping CSV file
                                Example: "Code_Machine_learning_Robotics/Trial_001/Results/filtered_final_frame_force_mapping.csv"
                                Or: "Code_Machine_learning_Robotics/Trial_001/Results/unfiltered_final_frame_force_mapping.csv"
        save_path (str, optional): Where to save the plot image
                                 Example: "Code_Machine_learning_Robotics/Trial_001/Results/force_plot.png"
                                 If None, displays plot interactively
    
    Expected CSV format:
        Columns: [Image_Number, Image_Timestamp, Force_Timestamp, Fx, Fy, Fz, Mx, My, Mz]
        Optional: [Ft] (tangential force, computed by filtering module)
    
    Plot features:
        - All force/torque channels shown as absolute values
        - Fz (normal force) highlighted in solid blue line
        - Tangential force (if available) shown as dashed blue line
        - Dual x-axes for time and frame correlation
        - Grid and legends for clarity
    """
    # Load the final mapping CSV
    df = pd.read_csv(final_mapping_path)
    
    # Convert all force/torque values to absolute values for clearer visualization
    for col in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
        if col in df.columns:
            df[col] = np.abs(df[col])

    # Compute tangential force if not already present
    if "F_tangential" not in df.columns and "Fx" in df.columns and "Fy" in df.columns:
        df["F_tangential"] = np.sqrt(df["Fx"]**2 + df["Fy"]**2)

    # Extract time and frame data for dual x-axes
    time = df["Image_Timestamp"]
    frames = df["Image_Number"]

    # Create the main plot with time on bottom x-axis
    fig, ax_time = plt.subplots(figsize=(10, 6))

    # Plot all 6 force/torque channels in light gray (background context)
    for col in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
        if col in df.columns:
            ax_time.plot(time, df[col], color="lightgray", linewidth=1)

    # Highlight the most important force components in bold blue
    if "Fz" in df.columns:
        ax_time.plot(time, df["Fz"], label="Fz (Normal Force)", color="blue", linewidth=2.5)
    
    if "F_tangential" in df.columns:
        ax_time.plot(time, df["F_tangential"], label="Tangential Force (√Fx² + Fy²)", 
                    color="blue", linestyle="--", linewidth=2.5)

    # Configure bottom x-axis (time) and y-axis
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Force [N] / Moment [Nm]")
    ax_time.set_title("Forces Over Time and Frame Index")
    ax_time.grid(True)
    ax_time.legend()

    # Add second x-axis for frame numbers (top of plot)
    ax_frame = ax_time.twiny()
    ax_frame.set_xlim(ax_time.get_xlim())
    ax_frame.set_xticks(ax_time.get_xticks())
    
    # Interpolate frame numbers corresponding to time tick positions
    frame_labels = np.interp(ax_time.get_xticks(), time, frames)
    ax_frame.set_xticklabels([f"{int(f)}" for f in frame_labels])
    ax_frame.set_xlabel("Frame Number")

    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Dual-axis plot saved to: {save_path}")
    else:
        plt.show()
