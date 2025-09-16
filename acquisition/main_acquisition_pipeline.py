
"""
Complete Tactile Sensing Data Acquisition Pipeline

This is the MAIN SCRIPT that orchestrates the entire tactile sensing data acquisition
and processing workflow. Run this file to execute a complete experiment from setup
to final processed dataset ready for machine learning.

🔄 COMPLETE WORKFLOW:
1. 📋 Interactive measurement setup (positions, indentor, sequence)
2. 🎯 Manual acquisition trigger (mouse click when ready)
3. 📊 Synchronized force sensor + camera data collection
4. 💾 Organized data saving with proper file structure  
5. 🔍 Quality diagnostics (dropped frames, missed samples)
6. 🔗 Force-to-image mapping for ML dataset creation
7. 🎚️ Optional signal filtering for noise reduction
8. 📍 Spatial position labeling for each image
9. 📈 Data visualization and verification plots
10. 🎥 Optional video creation for presentations

🗂️ DIRECTORY STRUCTURE CREATED:
Your base_directory/Trial_XXX/ will contain:
├── measurement_setup_data.csv      (experimental parameters)
├── Images/                         (image_1.tiff, image_2.tiff, ...)
├── force.csv                       (6-axis force measurements)
├── image_timestamps.csv            (image timing data)
├── force_timestamps.csv            (force timing data)
├── data_info.txt                   (acquisition metadata)
└── Results/                        (processed datasets)
    ├── positions_indentation.csv   (spatial labels)
    ├── unfiltered_final_frame_force_mapping.csv  (raw ML dataset)
    ├── filtered_final_frame_force_mapping.csv    (cleaned ML dataset)
    ├── highlighted_forces_dual_axis.png          (visualization)
    └── [Filtered/Unfiltered]_Force_Video.mp4     (optional video)

⚙️ CONFIGURATION:
Before running, configure the parameters in the "User-Defined Parameters" section:
- acquisition_time: How long to collect data (seconds)
- base_directory: Where to save your trial folders
- Force sensor settings: frequency, buffer size, calibration matrix
- Camera settings: frame rate, max allowed dropped frames

🚀 USAGE:
1. Connect your NI-DAQ force sensor and Basler camera
2. Update the configuration parameters below
3. Run this script: python main_acquisition_pipeline.py
4. Follow the interactive prompts for setup
5. Click mouse when ready to start acquisition
6. Wait for completion and review results

📋 REQUIREMENTS:
- NI-DAQ device with force sensor
- Basler camera 
- Python packages: nidaqmx, pypylon, opencv, pandas, matplotlib, scipy
- All acquisition modules (_00 through _07) in same directory
"""

import os
import time
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader
import threading
import matplotlib.pyplot as plt
from pynput.mouse import Listener
import pandas as pd
from pypylon import pylon
import cv2
import gc; gc.collect()

from _00_measurement_setup import setup_measurement_extended, generate_frame_positions, plot_force_and_position
from _01_acquisition import ForceAcquisition, CameraAcquisition, wait_for_click 
from _02_saving_data import save_acquisition_results
from _03_analysis import run_callback_and_sample_diagnostics
from _04_force_to_image_mapping import create_location_image_loss_file, create_final_mapping_file
from _05_low_pass_filter import butter_lowpass_filter, filter_final_frame_force_mapping
from _06_plotting import plot_final_mapping_forces
from _07_video_creation import create_video


if __name__ == "__main__":
	# =====================================================================================
	# 🛠️ USER-DEFINED PARAMETERS - CONFIGURE BEFORE RUNNING
	# =====================================================================================
	"""
	📝 PARAMETER EXPLANATION:

	1. ACQUISITION_TIME: Duration of data collection in seconds
	   - Longer time = more training data, but larger file sizes
	   - Typical range: 10-60 seconds depending on experiment
	   - Example: 30 seconds at 5000Hz = 150,000 force samples

	2. BASE_DIRECTORY: Root folder where all trial data will be saved
	   - Each trial creates a numbered subfolder (Trial_0001, Trial_0002, etc.)
	   - Raw string (r"path") handles Windows backslashes correctly
	   - Example: r"C:\Experiments\TactileSensing\Data"

	3. FORCE SENSOR CONFIGURATION:
	   - force_acquisition_frequency: Measurement frequency in Hz (higher = more precise)
	   - buffer_size: Memory chunks for processing (affects callback frequency)
	   - calibration_matrix: 6x6 transformation from voltages to forces/torques
	     Format: [[Fx_cal], [Fy_cal], [Fz_cal], [Mx_cal], [My_cal], [Mz_cal]]

	4. CAMERA CONFIGURATION:
	   - camera_frame_rate: Images per second (typical: 200-1000 Hz)
	   - max_dropped_frames: Quality threshold (acquisition fails if exceeded)

	5. PROCESSING OPTIONS:
	   - Enable/disable each step (True/False)
	   - Customize workflow based on your analysis needs
	"""

	########## 🎯 DATA COLLECTION SETTINGS ##########
	acquisition_time = 15  # Duration in seconds for force + image recording
	
	########## 📁 FILE STORAGE LOCATION ##########
	base_directory = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\2. Data creation\data aqcuisition\1\Data"

	########## ⚡ FORCE SENSOR HARDWARE SETUP ##########
	force_acquisition_frequency = 5000  # Force measurements per second [Hz] - high precision
	buffer_size = 500  # Memory buffer chunks (creates callback every buffer_size/frequency seconds)

	# 🎯 6-AXIS FORCE SENSOR CALIBRATION MATRIX
	# This 6x6 matrix converts raw voltages to physical units (N, Nm)
	# Obtain from your force sensor manufacturer calibration certificate (.cal file)
	calibration_matrix = [
		[0.60172,  32.34179,  -0.80031,  -0.11378,  -0.96768,  -33.52024],  # Fx calibration
		[  0.24029, -18.55637,   2.28165,  37.81298,  -1.09629,  -19.15258], # Fy calibration
		[-18.62115,  -0.57098, -17.97504,  -0.28311, -18.06730,    0.17530], # Fz calibration
		[ 32.28632,   0.94218,  -0.36544,  -0.03942, -31.99659,    0.42840], # Mx calibration
		[-19.35743,  -0.56220,  36.25709,   0.40960, -18.63986,    0.24755], # My calibration
		[ -0.14083,  18.62257,   1.05280,  18.36560,   0.71844,   19.41494], # Mz calibration
	]
	# 📏 CORRESPONDING SCALE VECTOR (from sensor XML calibration file)
	# This vector scales the calibration matrix values - order must match matrix rows!
	scale_vector = np.array([
		5.1694179576476,    # Fx scaling factor
		5.1694179576476,    # Fy scaling factor  
		1.77487848937509,   # Fz scaling factor
		221.35974008553,    # Tx (torque X) scaling factor
		221.35974008553,    # Ty (torque Y) scaling factor
		211.642988806056    # Tz (torque Z) scaling factor
	])
	# Apply scaling to create final calibration matrix (element-wise division)
	scaled_matrix = calibration_matrix / scale_vector[:, np.newaxis]

	########## 📸 CAMERA ACQUISITION SETTINGS ##########
	acquisition_frequency_camera = 50  # Images per second [Hz] - adjust based on camera specs
	Max_Number_of_allowed_lost_frames = 55  # Quality threshold: acquisition fails if exceeded

	########## 🔄 PROCESSING PIPELINE CONTROLS ##########
	# Enable/disable each step of the workflow (True/False)
	run_measurement_setup = True     # Interactive position/indentor configuration
	run_acquisition = True           # Force + camera data collection  
	run_saving = True               # Organize and save all data files
	run_analysis = True             # Quality diagnostics and verification
	run_mapping = True              # Create force-to-image dataset for ML
	run_filtering = True            # Optional noise reduction (recommended)
	run_plotting = True             # Generate verification plots
	save_individual_plots = False   # Create separate plot for each measurement point
	run_video_creation = False      # Generate MP4 videos (time-intensive)

	print("🚀 Starting Tactile Sensing Acquisition Pipeline...")
	print(f"📊 Configured for {acquisition_time}s acquisition")
	print(f"⚡ Force: {force_acquisition_frequency}Hz | 📸 Camera: {acquisition_frequency_camera}Hz") 
	print(f"💾 Data will be saved to: {base_directory}")
	print("=" * 80)

	########## 🎯 HARDWARE INITIALIZATION ##########
	# Initialize camera acquisition system (Basler camera setup)
	camera_acquisition = CameraAcquisition(frequency=acquisition_frequency_camera)
	
	# Initialize and configure Force acquisition system (NI-DAQ + 6-axis force sensor)
	force_acquisition = ForceAcquisition(
		device="Dev1",                    # NI-DAQ device name (check NI MAX)
		channels="ai0:5",                 # 6 analog input channels for 6-axis sensor
		frequency=force_acquisition_frequency,  # Sampling rate in Hz
		buffer_size=buffer_size,          # Memory buffer for data chunks
		calibration_matrix=scaled_matrix  # Calibrated voltage-to-force conversion
	)

	########## 📋 STEP 1: MEASUREMENT SETUP ##########
	if run_measurement_setup:
		print("\n📋 STEP 1: Interactive Measurement Setup")
		# Prompt for trial number to create organized data structure
		trail = input("🔢 Enter trial number (e.g., 001, 002, Trial_A): ")
		trial_dir = os.path.join(base_directory, trail)
		if not os.path.exists(trial_dir):
			os.makedirs(trial_dir)
			print(f"📁 Created trial directory: {trial_dir}")
		
		# Interactive setup: positions, indentors, measurement sequence
		# This creates measurement_setup_data.csv with experimental parameters
		setup_measurement_extended(trial_dir)
		print("✅ Setup complete - parameters saved to measurement_setup_data.csv")
	
	########## 🎯 STEP 2: DATA ACQUISITION ##########
	if run_acquisition:
		print("\n🎯 STEP 2: Synchronized Force + Camera Data Acquisition")
		print("⏳ Click anywhere when ready to start data collection...")
		wait_for_click()  # Manual trigger for precise start timing
		
		try:
			# Synchronized start timestamp for force and camera systems
			common_start_time = time.perf_counter()
			force_acquisition.common_start_time = common_start_time
			camera_acquisition.common_start_time = common_start_time
			start_time = common_start_time
			print(f'🕐 Common start time: {common_start_time:.6f}s')
			
			# Start force sensor data collection (continuous background sampling)
			force_acquisition.start_acquisition()
			print("⚡ Force acquisition started")
			
			# Start camera acquisition (triggered image capture for specified duration)
			image_data, timestamps, first_image_time, lost_frame_indices = camera_acquisition.run(
				acquisition_time=acquisition_time
			)
			print("📸 Camera acquisition completed")
			
		except Exception as e:
			print(f"❌ Error during acquisition: {e}")
		finally:
			# Ensure force acquisition stops even if camera fails
			force_acquisition.stop_acquisition()
			print("🛑 Force acquisition stopped")
			
			# Calculate total elapsed time and analyze acquisition quality
			elapsed_time = time.perf_counter() - start_time
			lost_frames = camera_acquisition.analyze_results(
				acquisition_time=acquisition_time, 
				elapsed_time=elapsed_time
			)
			
			# Process raw force data into calibrated forces
			raw_data, forces, timestamps_force = force_acquisition.process_data()
			print(f"✅ Acquisition complete: {elapsed_time:.2f}s total time")
			print(f"📊 Collected {len(forces)} force samples, {len(image_data)} images")
			
			# Quality control: Check for excessive frame loss
			if lost_frames > Max_Number_of_allowed_lost_frames:
				print(f"⚠️  WARNING: {lost_frames} frames lost (max allowed: {Max_Number_of_allowed_lost_frames})")
			else:
				print(f"✅ Frame loss acceptable: {lost_frames} frames lost")

	########## 🔍 STEP 3: QUALITY DIAGNOSTICS ##########
	if run_analysis:
		print("\n🔍 STEP 3: Data Quality Analysis & Diagnostics")
		# Analyze timing precision and data integrity - check for missed callbacks and lost samples
		callback_missed, force_sample_loss = run_callback_and_sample_diagnostics(
			acquisition=force_acquisition,
			raw_data=raw_data,
			buffer_size=buffer_size,
			force_acquisition_frequency=force_acquisition_frequency,
			acquisition_time=acquisition_time
		)
		print(f"📊 Timing Analysis: {callback_missed} missed callbacks, {force_sample_loss:.3f}% sample loss")
		
		# Create precise timestamps for every individual force sample (not just callbacks)
		# Each callback contains buffer_size samples, so we interpolate timestamps
		timestamps_force = np.array(timestamps_force)  # Callback timestamps from force acquisition
		samples_per_callback = buffer_size
		sampling_interval = 1.0 / force_acquisition_frequency  # Time between individual samples
		sample_timestamps = []
		
		for t_start in timestamps_force:
			# For each callback, create timestamps for all samples in that buffer
			chunk_times = t_start + np.arange(samples_per_callback) * sampling_interval
			sample_timestamps.extend(chunk_times)
		
		sample_timestamps = np.array(sample_timestamps)
		
		# Validate timestamp-force data alignment
		if sample_timestamps.shape[0] != forces.shape[0]:
			raise ValueError(f"❌ Mismatch between force samples and timestamps: "
							f"{sample_timestamps.shape[0]} timestamps vs {forces.shape[0]} force samples.")
		
		# Save continuous force timestamps for temporal analysis
		timestamps_force_path = os.path.join(trial_dir, "timestamps_continuous_forces.csv")
		pd.DataFrame(sample_timestamps, columns=["Timestamp"]).to_csv(timestamps_force_path, index=False)
		print(f"✅ Force sample timestamps saved to: {timestamps_force_path}")

	########## 💾 STEP 4: DATA SAVING & ORGANIZATION ##########
	if run_saving:
		print("\n💾 STEP 4: Organizing and Saving All Acquisition Data")
		# Only save data if quality standards are met
		if (lost_frames <= Max_Number_of_allowed_lost_frames):
			save_acquisition_results(
				image_data=image_data,
				timestamps=timestamps,
				timestamps_force=timestamps_force,
				forces=forces,
				force_acquisition_frequency=force_acquisition_frequency,
				buffer_size=buffer_size,
				acquisition_frequency_camera=acquisition_frequency_camera,
				elapsed_time=elapsed_time,
				trial_dir=trial_dir
			)
			print("✅ All data successfully saved with organized file structure")
		else:
			print("❌ ACQUISITION NOT SUCCESSFUL - Quality threshold exceeded, data not saved")
			print(f"   Lost frames: {lost_frames} > Max allowed: {Max_Number_of_allowed_lost_frames}")

	########## 🔗 STEP 5: FORCE-TO-IMAGE MAPPING ##########
	if run_mapping:
		print("\n🔗 STEP 5: Creating Force-to-Image Dataset for Machine Learning")
		# Create comprehensive mapping between images and corresponding force measurements
		mapping_filepath = os.path.join(trial_dir, "location_image_loss.csv")
		mapping_df = create_location_image_loss_file(
			mapping_filepath,
			acquisition_time=acquisition_time,
			camera_frequency=acquisition_frequency_camera,
			camera_timestamps=timestamps,
			force_timestamps=timestamps_force,
			force_frequency=force_acquisition_frequency,  
			tolerance_factor=0.5  # Temporal matching tolerance (0.5 = ±50% of camera period)
		)
		print("✅ Initial temporal mapping created - correlating images with forces")
		
		# Create comprehensive ML-ready dataset with exact force-image correspondence
		results_dir = os.path.join(trial_dir, "Results")
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
			print(f"📁 Created results directory: {results_dir}")
		
		# Define file paths for mapping process
		forces_csv_path = os.path.join(trial_dir, 'force.csv')
		final_mapping_filepath = os.path.join(results_dir, 'unfiltered_final_frame_force_mapping.csv')
		image_folder = os.path.join(trial_dir, 'Images')
		timestamps_force_path = os.path.join(trial_dir, "timestamps_continuous_forces.csv")
		image_timestamps_path = os.path.join(trial_dir, "image_timestamps.csv")
		
		# Create final ML dataset: each row = 1 image + corresponding 6-axis forces
		final_mapping_df = create_final_mapping_file(
			image_timestamps_path=image_timestamps_path,
			timestamps_force_path=timestamps_force_path,
			forces_csv_path=forces_csv_path,
			output_csv_path=final_mapping_filepath,
			image_folder=image_folder
		)
		print("✅ Final ML dataset created - unfiltered_final_frame_force_mapping.csv")

	########## 🎚️ STEP 6: SIGNAL FILTERING (OPTIONAL) ##########
	if run_filtering:
		print("\n🎚️ STEP 6: Signal Filtering for Noise Reduction")
		# Apply low-pass Butterworth filter to reduce sensor noise
		filtered_output_path = os.path.join(results_dir, "filtered_final_frame_force_mapping.csv")
		filter_final_frame_force_mapping(
			mapping_path=final_mapping_filepath,
			output_folder=results_dir,
			cutoff_ratio=0.1,  # Cutoff frequency as ratio of Nyquist frequency
			order=4            # Filter order (higher = sharper cutoff)
		)
		print("✅ Signal filtering complete - filtered_final_frame_force_mapping.csv")

	########## 📍 STEP 7: SPATIAL POSITION LABELING ##########
	print("\n📍 STEP 7: Spatial Position Labeling & Diagnostics")
	# Generate spatial coordinates for each measurement based on setup parameters
	generate_frame_positions(trial_dir)
	print("✅ Position labels generated based on measurement setup")
	
	# Create diagnostic plots showing force evolution and spatial positions  
	plot_force_and_position(trial_dir)
	print("✅ Position-force diagnostic plots created")

	########## 📈 STEP 8: DATA VISUALIZATION ##########
	if run_plotting:
		print("\n📈 STEP 8: Creating Data Verification Plots")
		# Generate comprehensive force visualization with highlighting
		plot_save_path = os.path.join(results_dir, "highlighted_forces_dual_axis.png")
		
		# Use filtered data if available, otherwise use unfiltered
		plot_data_path = filtered_output_path if run_filtering else final_mapping_filepath
		
		plot_final_mapping_forces(
			final_mapping_path=plot_data_path,
			save_path=plot_save_path,
			save_individual_plots=save_individual_plots  # Create separate plots per measurement
		)
		plt.show()  # Display interactive plot for immediate verification
		print(f"✅ Force visualization saved to: {plot_save_path}")

	########## 🎥 STEP 9: VIDEO CREATION (OPTIONAL) ##########
	if run_video_creation:
		print("\n🎥 STEP 9: Video Generation for Presentation & Analysis")
		create_vid = input("🎞️ Do you want to create a video? (y/n): ").strip().lower()
		if create_vid == "y":
			print("🎬 Generating video visualization...")
			print("   This combines force data with image sequence to create annotated MP4")
			print("   Video will show force evolution synchronized with tactile images")
			create_video(base_directory, trail)
			print("✅ Video creation completed")
		else:
			print("⏭️  Video creation skipped")

	########## 🎉 PIPELINE COMPLETION SUMMARY ##########
	print("\n" + "=" * 80)
	print("🎉 TACTILE SENSING ACQUISITION PIPELINE COMPLETED SUCCESSFULLY!")
	print("=" * 80)
	print(f"📁 Trial Directory: {trial_dir}")
	print(f"📊 Data Summary:")
	print(f"   • Force samples: {len(forces)} at {force_acquisition_frequency}Hz")
	print(f"   • Images captured: {len(image_data)} at {acquisition_frequency_camera}Hz") 
	print(f"   • Acquisition duration: {acquisition_time}s")
	print(f"   • Frame loss: {lost_frames}/{Max_Number_of_allowed_lost_frames} allowed")
	print(f"\n📄 Generated Files:")
	print(f"   • measurement_setup_data.csv    (experiment parameters)")
	print(f"   • force.csv                     (6-axis force measurements)")
	print(f"   • Images/                       (captured tactile images)")
	print(f"   • Results/                      (processed ML datasets)")
	print(f"     ├── unfiltered_final_frame_force_mapping.csv")
	if run_filtering:
		print(f"     ├── filtered_final_frame_force_mapping.csv")
	print(f"     ├── positions_indentation.csv")
	print(f"     └── highlighted_forces_dual_axis.png")
	print(f"\n🚀 Next Steps:")
	print(f"   1. Review force visualization plots for data quality")
	print(f"   2. Use ML dataset files for training tactile sensing models")
	print(f"   3. Analyze spatial position patterns in Results/ folder")
	print(f"   4. Consider running additional trials with different parameters")
	print("=" * 80)
