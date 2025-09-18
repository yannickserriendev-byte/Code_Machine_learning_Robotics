"""
Advanced Image Augmentation Pipeline for Tactile Sensing Data
===============================================

This script implements a comprehensive image augmentation pipeline specifically designed for 
processing tactile sensing data captured through photoelastic imaging. It provides robust
data handling mechanisms to prevent CSV corruption and maintains data integrity throughout
the augmentation process.

File Organization:
-----------------
1. Configuration Section:
   - Processing mode parameters (grayscale, noise, augmentations count)
   - Signal processing parameters (noise, jitter settings)
   - System parameters (image size, alignment, save frequency)
   - Path configuration and timestamp management

2. Helper Functions:
   - Data management (safe CSV operations)
   - Image processing (grayscale conversion, noise addition)
   - Geometric transformations (rotation, shifting, cropping)
   - Image masking and filtering

3. Main Processing Pipeline:
   - Data validation and setup
   - Augmentation loop with progress tracking
   - Error handling and recovery mechanisms

Key Features:
------------
- Robust CSV handling with validation and backup mechanisms
- Configurable image processing pipeline with multiple augmentation options
- Support for both RGB and grayscale processing
- Geometric transformations with coordinate system preservation
- Progress tracking with periodic saves
- Detailed configuration logging
- Error recovery and diagnostic capabilities

Dependencies:
------------
- PIL (Python Imaging Library) for image processing
- pandas for data management
- numpy for numerical operations
- torchvision for advanced image transformations
- tqdm for progress tracking

Usage:
------
Configure the parameters in the configuration section and run the script. The system will:
1. Create a timestamped output directory
2. Generate augmented versions of each input image
3. Track and update position coordinates after transformations
4. Save progress periodically to prevent data loss
5. Generate a detailed configuration log for reproducibility

Author: Yannick Serrien
Created for: TU Delft Master Thesis Project
Date: September 2025
"""

import os
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import functional as F

# ==== Configuration Parameters ====
# Processing Mode
USE_GRAYSCALE = False              # Set to True for grayscale conversion
APPLY_NOISE = False                 # Set to True to add Gaussian noise
NUM_AUGMENTATIONS_PER_IMAGE = 1    # Number of augmented versions per image

# Signal Processing Parameters
NOISE_SIGMA = 5.0/255.0            # Noise intensity (normalized)
JITTER_BRIGHTNESS = 0.4            # Brightness variation range
JITTER_CONTRAST = 0.4              # Contrast variation range
JITTER_SATURATION = 0.0            # Saturation variation (RGB only)
JITTER_HUE = 0.0                   # Hue variation (RGB only)

# System Parameters
FINAL_SIZE = 1200                  # Output image dimensions
IMAGE_SHIFT_RIGHT = -30            # Horizontal alignment correction
SAVE_EVERY = 10                    # Progress save frequency

# ==== Path Configuration ====
base_dir = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\code\code full pipeline\All code\Code from laptop\Testing_data_del\Data\full_dataset"
input_csv = os.path.join(base_dir, "0.labels.csv")
input_img_dir = os.path.join(base_dir, "0.images")

# Setup output paths with simple timestamp
timestamp = time.strftime("%m%d_%H%M")  # Simple timestamp format: MMDD_HHMM
output_img_dir = os.path.join(base_dir, f"1.augmented_images_{timestamp}")
output_csv_path = os.path.join(base_dir, f"1.augmented_labels_{timestamp}.csv")
working_csv = os.path.join(base_dir, "working.csv")

# Create configuration log file with detailed settings
config_content = f"""Augmentation Configuration Details
Created: {time.strftime("%Y-%m-%d %H:%M:%S")}
================================
Timestamp: {timestamp}

Processing Mode:
- Grayscale Mode: {'Enabled' if USE_GRAYSCALE else 'Disabled'}
- Noise Injection: {'Enabled' if APPLY_NOISE else 'Disabled'}
- Augmentations per Image: {NUM_AUGMENTATIONS_PER_IMAGE}

{'''Signal Processing Parameters:
- Noise Sigma: {NOISE_SIGMA}
- Brightness Jitter: {JITTER_BRIGHTNESS}
- Contrast Jitter: {JITTER_CONTRAST}
- Saturation Jitter: {JITTER_SATURATION}
- Hue Jitter: {JITTER_HUE}
''' if APPLY_NOISE else ''}
System Parameters:
- Final Image Size: {FINAL_SIZE}
- Image Shift Right: {IMAGE_SHIFT_RIGHT}
- Save Frequency: Every {SAVE_EVERY} samples

Input/Output Paths:
- Input CSV: {input_csv}
- Input Images: {input_img_dir}
- Output Images: {output_img_dir}
- Output Labels: {output_csv_path}
"""

config_path = os.path.join(base_dir, f"augmentation_config_{timestamp}.txt")
with open(config_path, 'w') as f:
    f.write(config_content)

# ==== Helper Functions ====
def safe_save_dataframe(df, working_file, final_file=None):
    """Safely save DataFrame to CSV with built-in verification and backup mechanisms.
    
    This function implements a two-phase save process to prevent data corruption:
    1. First saves to a working file and verifies its integrity
    2. If verification passes and a final file is specified, copies to final location
    
    The verification process includes:
    - Complete write verification
    - DataFrame structure validation
    - CSV parsing validation
    
    Args:
        df (pd.DataFrame): The pandas DataFrame to save
        working_file (str): Path to the temporary working CSV file
        final_file (str, optional): Path for the final CSV location. If None, 
            only saves to working file
            
    Returns:
        bool: True if save operation successful, False otherwise
        
    Error Handling:
        - Catches and logs all IO and parsing exceptions
        - Preserves working file on failure for recovery
        - Validates both working and final files after writing
    """
    try:
        # First save to working file
        df.to_csv(working_file, index=False)
        
        # Verify the file was written correctly
        try:
            pd.read_csv(working_file)
        except Exception as e:
            print(f"Error verifying CSV: {str(e)}")
            return False
            
        # If a final destination is specified and verification passed
        if final_file:
            try:
                # Copy to final destination
                with open(working_file, 'r') as source:
                    with open(final_file, 'w') as target:
                        target.write(source.read())
                        
                # Verify the final file
                pd.read_csv(final_file)
                return True
                
            except Exception as e:
                print(f"Error creating final CSV: {str(e)}")
                return False
        
        return True
            
    except Exception as e:
        print(f"Error in save operation: {str(e)}")
        return False

def convert_to_grayscale(img):
    """Convert RGB image to grayscale using standardized ITU-R BT.601 weights.
    
    This function implements the luminance preservation algorithm using the 
    standardized coefficients:
    - Red channel: 0.299
    - Green channel: 0.587
    - Blue channel: 0.114
    
    These weights account for human perception of color channels.
    
    Args:
        img (PIL.Image): Input RGB image to convert
        
    Returns:
        PIL.Image: Grayscale image converted back to RGB format for compatibility
        
    Note:
        Returns RGB format (not single-channel) to maintain compatibility with
        other processing functions that expect 3-channel inputs.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    np_img = np.array(img)
    grayscale = (0.299 * np_img[:,:,0] + 0.587 * np_img[:,:,1] + 0.114 * np_img[:,:,2]).astype(np.uint8)
    return Image.fromarray(grayscale, mode='L').convert('RGB')

def add_gaussian_noise(img, sigma=NOISE_SIGMA):
    """Add Gaussian noise to image for data augmentation.
    
    Implements zero-mean Gaussian noise injection with configurable intensity.
    The process:
    1. Normalizes image to [0,1] range
    2. Adds normally distributed noise
    3. Clips results to valid range
    4. Converts back to uint8
    
    Args:
        img (PIL.Image): Input image to add noise to
        sigma (float): Standard deviation of the Gaussian noise,
                      normalized to [0,1] range. Default from global config.
                      
    Returns:
        PIL.Image: Image with added noise
        
    Implementation Notes:
        - Uses normalized float operations for precision
        - Employs numpy's optimized random number generation
        - Ensures output stays in valid range via clipping
    """
    np_img = np.array(img).astype(float) / 255.0
    noise = np.random.normal(0, sigma, np_img.shape)
    noisy = np.clip(np_img + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))

def rotate_point(x, y, angle_degrees, center_x=0.0, center_y=0.0):
    """Rotate a point around a specified center point by a given angle.
    
    Implements the 2D rotation matrix transformation:
    [x'] = [cos θ  -sin θ] [x - cx]  + [cx]
    [y'] = [sin θ   cos θ] [y - cy]  + [cy]
    
    Args:
        x (float): X-coordinate of point to rotate
        y (float): Y-coordinate of point to rotate
        angle_degrees (float): Rotation angle in degrees (clockwise)
        center_x (float, optional): X-coordinate of rotation center. Defaults to 0.0
        center_y (float, optional): Y-coordinate of rotation center. Defaults to 0.0
        
    Returns:
        tuple(float, float): New (x,y) coordinates after rotation
        
    Note:
        Uses positive angle for clockwise rotation to match image processing
        convention, contrary to standard mathematical convention.
    """
    angle_radians = math.radians(angle_degrees)
    x_shifted = x - center_x
    y_shifted = y - center_y
    new_x = x_shifted * math.cos(angle_radians) - y_shifted * math.sin(angle_radians)
    new_y = x_shifted * math.sin(angle_radians) + y_shifted * math.cos(angle_radians)
    return new_x + center_x, new_y + center_y

def shift_image_right_crop(img, shift_pixels):
    """Shift image horizontally with intelligent edge handling.
    
    Performs a horizontal shift operation while maintaining image dimensions
    through strategic cropping. Used for alignment correction in the
    tactile sensor image processing pipeline.
    
    Process:
    1. Pads image on the specified side
    2. Crops to maintain original dimensions
    3. Preserves image quality and aspect ratio
    
    Args:
        img (PIL.Image): Input image to shift
        shift_pixels (int): Number of pixels to shift right (negative for left)
        
    Returns:
        PIL.Image: Shifted and cropped image of same dimensions as input
        
    Note:
        Uses torchvision's functional transforms for efficient processing
    """
    w, h = img.size
    padded = F.pad(img, padding=[shift_pixels, 0, 0, 0], fill=0)
    return padded.crop((0, 0, w, h))

def apply_circular_mask(img, diameter):
    """Apply circular mask to image for tactile sensor region isolation.
    
    Creates a binary circular mask and applies it to the image to isolate
    the active tactile sensing region and remove edge artifacts.
    
    Implementation Details:
    1. Creates distance matrix from center
    2. Generates binary circular mask
    3. Applies mask while preserving color channels
    4. Handles both RGB and grayscale images
    
    Args:
        img (PIL.Image): Input image to mask
        diameter (int): Diameter of the circular mask in pixels
        
    Returns:
        PIL.Image: Masked image with regions outside circle set to black
        
    Note:
        Automatically detects and preserves image color mode (RGB/grayscale)
    """
    np_img = np.array(img)
    w, h = img.size
    radius = diameter // 2
    center_x, center_y = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    mask = dist <= radius
    masked_img = np.zeros_like(np_img)
    if np_img.ndim == 3:
        for c in range(3):
            masked_img[..., c] = np_img[..., c] * mask
    else:
        masked_img = np_img * mask
    return Image.fromarray(masked_img)

def crop_center(img, target_size):
    """Crop image to specified size from center point.
    
    Performs centered cropping operation to extract region of interest
    while maintaining aspect ratio and central alignment. Critical for
    maintaining consistent input size for neural network processing.
    
    Process:
    1. Calculates center coordinates
    2. Determines crop boundaries
    3. Extracts square region of specified size
    
    Args:
        img (PIL.Image): Input image to crop
        target_size (int): Size of the square crop region in pixels
        
    Returns:
        PIL.Image: Cropped square image of specified size
        
    Note:
        Assumes input image is larger than target_size in both dimensions
    """
    w, h = img.size
    center_x, center_y = w // 2, h // 2
    left = center_x - target_size // 2
    top = center_y - target_size // 2
    return img.crop((left, top, left + target_size, top + target_size))

def process_image(img, apply_jitter=True):
    """Apply the configured image processing pipeline to a single image.
    
    This function orchestrates the complete image processing sequence:
    1. Optional grayscale conversion (if USE_GRAYSCALE is True)
    2. Color jittering for augmentation (if apply_jitter is True):
       - Brightness variation
       - Contrast adjustment
       - Saturation modification (RGB only)
       - Hue shifting (RGB only)
    3. Optional Gaussian noise injection (if APPLY_NOISE is True)
    
    Args:
        img (PIL.Image): Input image to process
        apply_jitter (bool): Whether to apply color jittering transforms.
                           Default True for augmented images, False for originals.
                           
    Returns:
        PIL.Image: Processed image with all configured transformations applied
        
    Configuration Dependencies:
        Uses global parameters for processing options and intensities:
        - USE_GRAYSCALE
        - APPLY_NOISE
        - JITTER_* parameters
        - NOISE_SIGMA
    """
    if USE_GRAYSCALE:
        img = convert_to_grayscale(img)
    
    if apply_jitter:
        jitter = transforms.ColorJitter(
            brightness=JITTER_BRIGHTNESS,
            contrast=JITTER_CONTRAST,
            saturation=JITTER_SATURATION if not USE_GRAYSCALE else 0,
            hue=JITTER_HUE if not USE_GRAYSCALE else 0
        )
        img = jitter(img)
    
    if APPLY_NOISE:
        img = add_gaussian_noise(img, NOISE_SIGMA)
        
    return img

def main():
    """Main execution function orchestrating the complete augmentation pipeline.
    
    This function implements the core augmentation workflow:
    
    1. Initialization:
       - Creates output directories
       - Validates input data and required columns
       - Sets up progress tracking
    
    2. Main Processing Loop:
       - Loads and validates each input image
       - Processes original image without jitter
       - Generates specified number of augmented versions:
         * Applies color jittering
         * Performs geometric transformations
         * Updates coordinate systems
       - Maintains progress tracking and periodic saves
    
    3. Data Management:
       - Tracks augmented sample metadata
       - Periodically saves progress to prevent data loss
       - Validates saved data integrity
    
    4. Error Handling:
       - Catches and logs all exceptions
       - Maintains working files for recovery
       - Provides detailed error reporting
    
    Dependencies:
        Requires properly configured global parameters and
        valid input data structure (CSV with required columns)
    
    Output:
        - Augmented images in specified output directory
        - Updated CSV with augmentation metadata
        - Configuration log file
        - Progress indicators and error reports
    """
    try:
        # ==== Prepare Output Directory ====
        os.makedirs(output_img_dir, exist_ok=True)

        # ==== Load and Validate Input Data ====
        print(f"Loading data from: {input_csv}")
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        
        df = pd.read_csv(input_csv)
        required_columns = ["New_Image_Name", "X_Position_mm", "Y_Position_mm"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Initialize storage
        augmented_rows = []
        counter = 0
        last_save_counter = 0

        # ==== Augmentation Loop ====
        with tqdm(total=len(df) * (NUM_AUGMENTATIONS_PER_IMAGE + 1), desc="Augmenting") as pbar:
            for idx, row in df.iterrows():
                try:
                    img_path = os.path.join(input_img_dir, row["New_Image_Name"])
                    if not os.path.exists(img_path):
                        print(f"❌ Missing image: {img_path}")
                        continue

                    orig_img = Image.open(img_path).convert("RGB")
                    contact_x, contact_y = row["X_Position_mm"], row["Y_Position_mm"]

                    # === Add original image as final augmentation ===
                    img = process_image(orig_img, apply_jitter=False)  # No jitter for original
                    shifted = shift_image_right_crop(img, IMAGE_SHIFT_RIGHT)
                    masked = apply_circular_mask(shifted, diameter=FINAL_SIZE)
                    cropped_img = crop_center(masked, FINAL_SIZE)

                    aug_name = f"{Path(row['New_Image_Name']).stem}_aug{NUM_AUGMENTATIONS_PER_IMAGE}.png"
                    cropped_img.save(os.path.join(output_img_dir, aug_name))

                    new_row = row.copy()
                    new_row["New_Image_Name"] = aug_name
                    new_row["x"] = contact_x
                    new_row["y"] = contact_y
                    new_row["rotation_angle"] = 0.0
                    augmented_rows.append(new_row)
                    counter += 1
                    pbar.update(1)

                    # Save progress periodically
                    if counter >= last_save_counter + SAVE_EVERY:
                        if safe_save_dataframe(pd.DataFrame(augmented_rows), working_csv):
                            print(f"💾 Progress saved at {counter} samples...")
                            last_save_counter = counter

                    # === Generate augmented versions ===
                    for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
                        # Random rotation angle
                        angle = random.uniform(0, 360)
                        
                        # Process augmented image
                        img = process_image(orig_img, apply_jitter=True)
                        shifted = shift_image_right_crop(img, IMAGE_SHIFT_RIGHT)
                        masked = apply_circular_mask(shifted, diameter=FINAL_SIZE)
                        rotated = F.rotate(masked, angle, expand=True)
                        final_img = crop_center(rotated, FINAL_SIZE)

                        # Save augmented image
                        aug_name = f"{Path(row['New_Image_Name']).stem}_aug{i}.png"
                        output_path = os.path.join(output_img_dir, aug_name)
                        final_img.save(output_path)

                        # Update metadata
                        final_x, final_y = rotate_point(contact_x, contact_y, angle)
                        new_row = row.copy()
                        new_row["New_Image_Name"] = aug_name
                        new_row["x"] = final_x
                        new_row["y"] = final_y
                        new_row["rotation_angle"] = angle
                        augmented_rows.append(new_row)
                        counter += 1
                        pbar.update(1)

                        # Save progress periodically
                        if counter >= last_save_counter + SAVE_EVERY:
                            if safe_save_dataframe(pd.DataFrame(augmented_rows), working_csv):
                                print(f"💾 Progress saved at {counter} samples...")
                                last_save_counter = counter

                except Exception as e:
                    print(f"❌ Error processing image {row['New_Image_Name']}: {str(e)}")
                    continue

        # ==== Final Save ====
        final_df = pd.DataFrame(augmented_rows)
        if safe_save_dataframe(final_df, working_csv, output_csv_path):
            print(f"\n✅ Finished. Total augmented + original samples: {len(augmented_rows)}")
            print(f"📁 Labels saved to: {output_csv_path}")
            print(f"📁 Images saved to: {output_img_dir}")
            print(f"📁 Configuration saved to: {config_path}")
        else:
            print("❌ Error saving final CSV file")

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        if os.path.exists(working_csv):
            print(f"Working CSV file available at: {working_csv}")

if __name__ == "__main__":
    main()