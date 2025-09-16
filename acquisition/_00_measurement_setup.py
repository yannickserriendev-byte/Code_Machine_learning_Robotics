
import os
import csv

"""
Measurement Setup and Position Calculation Module

This module handles the initial setup configuration for tactile sensing experiments.
It collects all measurement parameters from the user and calculates frame-wise positions
for each acquired image based on the experimental sequence.

Key functions:
    - setup_measurement_extended(): Interactive setup data collection
    - generate_frame_positions(): Calculate X,Y positions for each image frame
    - plot_force_and_position(): Visualize force vs position relationships

Typical workflow:
    1. Create your trial folder: Code_Machine_learning_Robotics/Trial_001/
    2. Run setup_measurement_extended() to record all experimental parameters
    3. After acquisition, run generate_frame_positions() to label each image with its position
    4. Optionally use plot_force_and_position() for visualization

This creates the foundation for your tactile sensing dataset with proper
spatial labeling of each image frame.

Expected directory structure:
    Code_Machine_learning_Robotics/
    └── Trial_001/ (your trial folder)
        ├── measurement_setup_data.csv (created by this module)
        ├── image_timestamps.csv (from acquisition)
        └── Results/
            └── positions_indentation.csv (positions for each frame)
"""

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def setup_measurement_extended(trial_dir):
    """
    Interactively collect and record all measurement setup parameters for a trial.
    
    This function guides you through setting up a new tactile sensing experiment by
    collecting all the necessary parameters about your hardware setup, indentor,
    and experimental sequence. It saves everything to a CSV file for later use.
    
    Parameters:
        trial_dir (str): Full path to your trial folder
                        Example: "C:/Code_Machine_learning_Robotics/Trial_001"
                        Or: "Code_Machine_learning_Robotics/Experiment_2024_03_15"
    
    Process:
        1. Prompts you for all setup parameters with clear explanations
        2. Shows a summary of your inputs for verification
        3. Saves parameters to measurement_setup_data.csv in your trial folder
        4. Repeats if you need to make corrections
    
    Required information:
        - Stepper motor positions (X-axis, ground reference and starting position)
        - Manual stage position (Y-axis, ground reference and current position)  
        - Indentor details (radius in mm, shape description)
        - Sequence timing (duration per step, sequence filename)
    
    Output file:
        Creates: trial_dir/measurement_setup_data.csv
        This file is used by generate_frame_positions() and the rest of the pipeline.
    """
    while True:
        # Ensure trial directory exists
        os.makedirs(trial_dir, exist_ok=True)
        setup_csv_path = os.path.join(trial_dir, "measurement_setup_data.csv")

        print("\n" + "="*60)
        print("🔧 TACTILE SENSING EXPERIMENT SETUP")
        print("="*60)
        print("Please provide the following information about your experimental setup.")
        print("All measurements should be in millimeters (mm).")
        print("")

        # Collect stepper motor positions with detailed explanations
        print("📍 STEPPER MOTOR SETUP (X-AXIS - Horizontal Movement)")
        print("The stepper motor moves your indentor horizontally across the sensor.")
        print("")
        x_ground = float(input("🏠 Enter the REFERENCE/ZERO X position (mm): "))
        print("   → This is your baseline position (often where you first contact the sensor)")
        print("")
        x_start = float(input("🚀 Enter the STARTING X position for this sequence (mm): "))
        print("   → This is where the indentor will begin before starting the movement sequence")
        print("")

        # Collect manual stage positions  
        print("📍 MANUAL STAGE SETUP (Y-AXIS - Perpendicular Movement)")
        print("The manual stage controls the perpendicular positioning of your setup.")
        print("")
        y_ground = float(input("🏠 Enter the REFERENCE/ZERO Y position (mm): "))
        print("   → This is your baseline Y position for reference")
        print("")
        y_current = float(input("📍 Enter the CURRENT Y position for this trial (mm): "))
        print("   → This is where the Y-axis is positioned for this specific experiment")
        print("")

        # Collect indentor specifications. This is to ensure you know and save what indentor was used to create the dataset.
        print("🔹 INDENTOR SPECIFICATIONS")
        print("Describe the physical properties of your indenting tool.")
        print("")
        indentor_radius = float(input("📏 Enter indentor radius (mm): "))
        print("")
        indentor_shape = input("🔺 Enter indentor shape (e.g., spherical, flat, conical, cylindrical): ")
        print("")

        # Collect sequence and timing information
        print("⏱️  EXPERIMENTAL SEQUENCE SETTINGS")
        print("Define the timing and control file for your experiment.")
        print("")
        indentation_duration = float(input("⏲️  Enter duration per indentation step (seconds): "))
        print("   → Include time for both indentation AND X-axis movement to next position")
        print("")
        sequence_filename = input("📄 Enter the sequence filename (e.g., 'indent_5mm_2s.sequence'): ")
        print("")
        print("⚠️  IMPORTANT: Make sure this sequence file exists in your trial folder!")
        print(f"   Expected location: {trial_dir}/{sequence_filename}")
        print("")

        # Save all setup parameters to CSV file
        with open(setup_csv_path, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Parameter", "Value"])
            writer.writerow(["X_Ground_Position_mm", x_ground])
            writer.writerow(["X_Start_Position_mm", x_start])
            writer.writerow(["Y_Ground_Position_mm", y_ground])
            writer.writerow(["Y_Current_Position_mm", y_current])
            writer.writerow(["Indentor_Radius_mm", indentor_radius])
            writer.writerow(["Indentor_Shape", indentor_shape])
            writer.writerow(["Indentation_Duration_s", indentation_duration])
            writer.writerow(["Sequence_File_Name", sequence_filename])

        # Display comprehensive summary for user verification
        df = pd.read_csv(setup_csv_path)
        print("="*60)
        print("📋 EXPERIMENT SETUP SUMMARY")
        print("="*60)
        print(df.to_string(index=False))
        print("")
        print("💾 Preview of saved file: measurement_setup_data.csv")
        print(f"📁 Location: {setup_csv_path}")
        print("")

        # Get user confirmation with clear options
        confirm = input("✅ Is all information correct? (y/n): ").strip().lower()
        if confirm == "y":
            print("")
            print("🎉 SUCCESS! Setup configuration saved successfully.")
            print(f"📁 File saved to: {setup_csv_path}")
            print("")
            print("🚀 NEXT STEPS:")
            print("   1. Run your acquisition experiment")
            print("   2. After acquisition, run generate_frame_positions() to calculate positions")
            print("")
            break
        else:
            print("")
            print("🔄 Let's correct the information. Please re-enter the setup details.")
            print("")

def generate_frame_positions(trial_dir):
    """
    Calculate X and Y positions for each acquired image frame based on setup parameters.
    
    This function takes the timestamps from your acquired images and the setup parameters
    to calculate where the indentor was positioned when each image was captured.
    This spatial labeling is essential for machine learning and analysis.
    
    Parameters:
        trial_dir (str): Path to your trial folder containing setup and acquisition data
                        Example: "Code_Machine_learning_Robotics/Trial_001"
    
    Required input files:
        - measurement_setup_data.csv (from setup_measurement_extended)
        - image_timestamps.csv (from image acquisition)
    
    Process:
        1. Loads your experimental setup parameters
        2. Reads image timestamps from acquisition
        3. Infers movement direction from sequence filename
        4. Calculates X,Y position for each image based on timing
        5. Saves position data as CSV for ML pipeline
    
    Algorithm:
        - Uses constant step size (5mm) and timing from setup
        - Calculates which indentation step each image belongs to
        - Converts absolute positions to relative positions from ground reference
        - Assumes linear movement with constant timing between steps
    
    Output:
        Creates: trial_dir/Results/positions_indentation.csv
        Columns: [Image_Number, Image_Timestamp, X_Position_mm, Y_Position_mm, Indentor_Shape]
        
    This file becomes part of your final ML dataset with spatial labels for each image.
    """
    setup_path = os.path.join(trial_dir, "measurement_setup_data.csv")
    setup = pd.read_csv(setup_path).set_index("Parameter")["Value"]

    image_ts_path = os.path.join(trial_dir, "image_timestamps.csv")
    image_ts = pd.read_csv(image_ts_path)["Image_timestamps"].to_numpy()

    # Extract setup parameters
    x_ground = float(setup["X_Ground_Position_mm"])
    x_start = float(setup["X_Start_Position_mm"])
    y_ground = float(setup["Y_Ground_Position_mm"])
    y_current = float(setup["Y_Current_Position_mm"])
    radius = float(setup["Indentor_Radius_mm"])
    indentor_shape = setup["Indentor_Shape"]
    duration = float(4.8)  # TODO: Replace with setup["Indentation_Duration_s"] if available

    sequence_filename = setup["Sequence_File_Name"]

    # Set constant step size (mm)
    x_step_mm = 5

    # Infer movement direction from sequence filename
    if "right" in sequence_filename.lower():
        direction = 1
    elif "left" in sequence_filename.lower():
        direction = -1
    else:
        print("[!] Could not determine direction from filename. Defaulting to right (+X).")
        direction = 1

    y_real = y_current - y_ground

    # Calculate real X positions for each frame
    x_real_values = []
    t0 = image_ts[0]
    for t in image_ts:
        step_idx = int((t - t0) // duration)
        x_pos = x_start + direction * step_idx * x_step_mm
        x_real = x_pos - x_ground 
        x_real_values.append(x_real)

    y_real_values = [y_real] * len(image_ts)
    indentor_shapes = [indentor_shape] * len(image_ts)

    # Save frame-wise positions to CSV
    df_out = pd.DataFrame({
        "Image_Number": np.arange(1, len(image_ts) + 1),
        "Image_Timestamp": image_ts,
        "X_Position_mm": x_real_values,
        "Y_Position_mm": y_real_values,
        "Indentor_Shape": indentor_shapes
    })

    output_path = os.path.join(trial_dir, "Results", "positions_indentation.csv")
    df_out.to_csv(output_path, index=False)
    print(f"[OK] Frame-wise positions saved to: {output_path}")

def plot_force_and_position(trial_dir):
    """
    Visualize the relationship between absolute normal force and X position over time.
    
    This function creates a diagnostic plot showing how the force measurements
    change as the indentor moves across different X positions. Very useful for
    validating your experimental data and understanding force-position relationships.
    
    Parameters:
        trial_dir (str): Path to your trial folder
                        Example: "Code_Machine_learning_Robotics/Trial_001"
    
    Required input files:
        - Results/positions_indentation.csv (from generate_frame_positions)
        - Results/unfiltered_final_frame_force_mapping.csv (from force-image mapping)
    
    Plot features:
        - X-axis: Image timestamp (time in seconds)
        - Y-axis: Force (N) and Position (mm) 
        - Blue line: Absolute normal force |Fz|
        - Green line: X position of indentor
        - Gray vertical lines: Mark each X position change
        - Grid and legends for clarity
    
    Use this to:
        - Verify your experimental sequence worked correctly
        - Identify any timing or positioning issues
        - Understand force patterns across different spatial locations
        - Validate data quality before machine learning
    
    The plot will display interactively - close it to continue your workflow.
    """
    # Load position and force data
    pos_path = os.path.join(trial_dir, "Results", "positions_indentation.csv")
    force_path = os.path.join(trial_dir, "Results", "unfiltered_final_frame_force_mapping.csv")

    pos_df = pd.read_csv(pos_path)
    force_df = pd.read_csv(force_path)

    # Merge on image timestamp
    merged = pd.merge(pos_df, force_df, on="Image_Timestamp", how="inner")

    # Calculate absolute Fz (normal force)
    merged["Abs_Fz"] = merged["Fz"].abs()

    # Identify X position changes
    position_changes = merged[merged["X_Position_mm"].diff().fillna(0) != 0]

    # Plot force and position
    plt.figure(figsize=(12, 6))
    plt.plot(merged["Image_Timestamp"], merged["Abs_Fz"], label="|Fz| (N)", color="blue")
    plt.plot(merged["Image_Timestamp"], merged["X_Position_mm"], label="X Position (mm)", color="green")

    # Add vertical lines at each X position change
    for ts in position_changes["Image_Timestamp"]:
        plt.axvline(x=ts, color="gray", linestyle="--", alpha=0.4)

    plt.xlabel("Image Timestamp (s)")
    plt.ylabel("Force (N) / Position (mm)")
    plt.title("Absolute Normal Force vs Time with X Position Overlaid")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage for new users:
    # 
    # Step 1: Define your trial folder path
    # trial_dir = r"C:\Code_Machine_learning_Robotics\Trial_001"
    # OR on other systems:
    # trial_dir = "Code_Machine_learning_Robotics/Trial_001"
    #
    # Step 2: Run initial setup (do this BEFORE acquisition)
    # setup_measurement_extended(trial_dir)
    # 
    # Step 3: [Run your acquisition experiment with main_acquisition_pipeline.py]
    #
    # Step 4: Calculate positions for each image (do this AFTER acquisition)
    # generate_frame_positions(trial_dir)
    #
    # Step 5: Optionally visualize results
    # plot_force_and_position(trial_dir)
    
    pass
