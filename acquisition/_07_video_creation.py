"""
Video Creation Module for Acquisition Results Visualization

This module creates synchronized videos combining acquired images with real-time force plots.
Perfect for presentations, debugging, and understanding the relationship between 
tactile images and force measurements over time.

Key features:
    - Side-by-side video: Original images + real-time force plot
    - User choice between filtered or unfiltered force data
    - Moving window force plot that follows current frame
    - Customizable y-axis limits and window size
    - MP4 output for easy sharing and presentation

Typical usage:
    1. After completing acquisition and creating final mapping CSV
    2. Call create_video() to generate visualization
    3. Choose filtered vs unfiltered data based on your analysis needs
    4. Use resulting video for presentations or analysis validation

Directory context:
    Input folder: Code_Machine_learning_Robotics/[trial_folder]/
        - Images/ (contains image_1.tiff, image_2.tiff, ...)
        - Results/ (contains filtered_final_frame_force_mapping.csv and unfiltered version)
    Output: Code_Machine_learning_Robotics/[trial_folder]/Results/[Filtered/Unfiltered]_Force_Video.mp4

Video structure:
    - Left side: Original tactile images (maintains original resolution)
    - Right side: Real-time force plot with sliding window
    - Force plot shows current frame position with vertical line
"""

def create_video(base_dir, trial_name):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import os
    from io import BytesIO
    from PIL import Image
    from tqdm import tqdm
    """
    Create a synchronized video combining tactile images with real-time force visualization.
    
    This function generates an MP4 video where each frame shows:
        - Left: The corresponding tactile image (image_N.tiff)
        - Right: Force plot with sliding window centered on current frame
        - Vertical line on force plot indicating current frame position
    
    Parameters:
        base_dir (str): Path to your main data directory
                       Example: "C:/Code_Machine_learning_Robotics" or just "Code_Machine_learning_Robotics"
        trial_name (str): Name of your specific trial folder
                        Example: "Trial_001" or "Experiment_2024_03_15"
    
    Expected folder structure:
        base_dir/
        └── trial_name/
            ├── Images/
            │   ├── image_1.tiff
            │   ├── image_2.tiff
            │   └── ...
            └── Results/
                ├── filtered_final_frame_force_mapping.csv
                ├── unfiltered_final_frame_force_mapping.csv
                └── [output videos will be saved here]
    
    Interactive prompts:
        - Choose filtered (f) or unfiltered (u) force data
        - Set Y-axis limits for force plot (or use automatic scaling)
    
    Output files:
        - Filtered_Force_Video.mp4 (if filtered data chosen)
        - Unfiltered_Force_Video.mp4 (if unfiltered data chosen)
    
    Video parameters:
        - Frame rate: 2 fps (adjustable in code)
        - Resolution: Matches original image resolution x2 (side-by-side)
        - Format: MP4 with H.264 codec
        - Window size: 100 frames visible in force plot (adjustable in code)
    """
    # Helper function to convert matplotlib figure to image array
    def fig_to_img(fig):
        """Convert matplotlib figure to numpy array for video frame."""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img = np.array(img.convert('RGB'))
        buf.close()
        plt.close(fig)
        return img

    # Define force labels with units for plot clarity
    LABELS_WITH_UNITS = {
        'Ft': 'Tangential Force (N)',
        'Normal (|Fz|)': 'Normal Force (|Fz|) (N)',
        'Fx': 'Fx (N)', 'Fy': 'Fy (N)', 'Fz': 'Fz (N)',
        'Mx': 'Mx (Nmm)', 'My': 'My (Nmm)', 'Mz': 'Mz (Nmm)'
    }

    # Set up directory paths based on your trial structure
    trial_dir = os.path.join(base_dir, trial_name)
    image_folder = os.path.join(trial_dir, "Images")
    results_dir = os.path.join(trial_dir, "Results")

    # Ask user to choose between filtered or unfiltered force data
    choice = input("▶️ Create video from (f)iltered or (u)nfiltered data? ").strip().lower()
    if choice == 'f':
        force_csv_path = os.path.join(results_dir, "filtered_final_frame_force_mapping.csv")
        output_video_path = os.path.join(results_dir, "Filtered_Force_Video.mp4")
    elif choice == 'u':
        force_csv_path = os.path.join(results_dir, "unfiltered_final_frame_force_mapping.csv")
        output_video_path = os.path.join(results_dir, "Unfiltered_Force_Video.mp4")
    else:
        print("❌ Invalid input. Skipping video creation.")
        return

    # Load the force mapping CSV
    try:
        df = pd.read_csv(force_csv_path)
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return

    # Compute derived force quantities if not present
    if 'Fx' in df.columns and 'Fy' in df.columns:
        df['Ft'] = np.sqrt(df['Fx']**2 + df['Fy']**2)
    if 'Fz' in df.columns:
        df['Normal (|Fz|)'] = np.abs(df['Fz'])

    # Identify available force columns for plotting
    force_columns = [col for col in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'] if col in df.columns]
    n_frames = len(df)
    image_numbers = np.arange(1, n_frames + 1)

    # Get user preferences for plot y-axis limits
    try:
        y_min = float(input("📉 Y-axis minimum (e.g., -10): "))
        y_max = float(input("📈 Y-axis maximum (e.g., 15): "))
    except ValueError:
        y_min = y_max = None
        print("⚠️ No valid y-axis bounds given. Using automatic scaling.")

    # Determine video resolution from first image
    example_image_path = os.path.join(image_folder, "image_1.tiff")
    img_example = cv2.imread(example_image_path)
    if img_example is None:
        raise RuntimeError(f"Could not load example image to determine size: {example_image_path}")

    img_height, img_width, _ = img_example.shape

    # Video configuration
    frame_width = img_width * 2  # Side-by-side: image + plot
    frame_height = img_height
    fps = 2  # Frames per second in output video
    window_size = 100  # Number of frames visible in force plot window

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("📹 Creating video...")
    for idx in tqdm(range(n_frames), desc="Generating frames"):
        # Generate current frame for video
        current_frame = idx + 1
        image_filename = os.path.join(image_folder, f"image_{current_frame}.tiff")
        img = cv2.imread(image_filename)
        if img is None:
            # Create blank image if file missing
            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # Define sliding window for force plot (centered on current frame)
        half_window = window_size // 2
        start_idx = max(0, idx - half_window)
        end_idx = min(n_frames, idx + half_window)
        frame_window = image_numbers[start_idx:end_idx]

        # Create force plot for current window
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Plot all force channels in light gray background
        for col in force_columns:
            ax.plot(frame_window, np.abs(df[col].iloc[start_idx:end_idx]),
                    color="lightgray", linewidth=1)

        # Highlight important force components
        if "Fz" in df.columns:
            ax.plot(frame_window, np.abs(df["Fz"].iloc[start_idx:end_idx]),
                    label="Fz (Normal Force)", color="blue", linewidth=2.5)

        if "Ft" in df.columns:
            ax.plot(frame_window, np.abs(df["Ft"].iloc[start_idx:end_idx]),
                    label="Tangential Force (√Fx² + Fy²)", color="blue", linestyle="--", linewidth=2.5)

        # Add vertical line showing current frame position
        ax.axvline(x=current_frame, color='k', linestyle='--', linewidth=2)
        ax.set_xlim(frame_window[0], frame_window[-1])
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Force [N] / Moment [Nm]")
        ax.set_title("6-Axial Force vs Frame Number")
        ax.grid(True)
        ax.legend()

        # Apply custom y-axis limits if specified
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        
        # Convert plot to image and resize to match original image height
        plot_img = fig_to_img(fig)
        plot_img = cv2.resize(plot_img, (img_width, img_height))

        # Combine original image and force plot side-by-side
        combined_frame = np.hstack((img, plot_img))
        out.write(combined_frame)

    # Finalize video
    out.release()
    print(f"\n✅ Video saved to: {output_video_path}")
