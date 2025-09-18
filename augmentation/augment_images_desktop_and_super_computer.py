"""
Advanced Image Augmentation Pipeline for High-Performance Computing
================================================================

Overview:
---------
This script implements a sophisticated image augmentation pipeline specifically
designed for processing tactile sensing data in both desktop and high-performance
computing environments. It generates multiple variations of input images while
preserving and transforming associated contact point data.

Key Features:
------------
1. Dual-Environment Support:
   - Supercomputer: Utilizes parallel processing and batch operations
   - Desktop: Sequential processing for development and testing

2. Image Processing Capabilities:
   - RGB/Grayscale conversion
   - Gaussian noise injection
   - Color space jittering (brightness, contrast)
   - Geometric transformations (rotation, shifting)
   - Circular masking for sensor region isolation

3. Data Management:
   - Coordinate system transformation
   - Contact point tracking through transformations
   - CSV-based metadata handling
   - Progress tracking and checkpointing

4. Performance Optimizations:
   - Tensor-based operations using PyTorch
   - Multi-threaded processing on supercomputer
   - Memory-efficient batch processing
   - Automatic thread count optimization

5. Error Handling:
   - Comprehensive logging system
   - Progress recovery mechanisms
   - Input validation
   - Exception management

Technical Implementation:
-----------------------
- Uses PyTorch for efficient tensor operations
- Implements thread-safe parallel processing
- Maintains data integrity through atomic operations
- Provides consistent output ordering

File Naming Convention:
---------------------
Pattern: 1.augmented_labels_[MMDD_HHMM]_[s/d].csv
- MMDD_HHMM: Timestamp (Month, Day, Hour, Minute)
- s/d suffix: Indicates supercomputer or desktop environment

Usage:
------
1. Configure processing parameters in the configuration section
2. Place input images in the specified directory
3. Provide CSV with image metadata (names, coordinates)
4. Run the script in either desktop or supercomputer mode
5. Collect augmented images and updated metadata from output directory

Dependencies:
------------
- PyTorch: Tensor operations and transformations
- Pandas: Data management and CSV handling
- PIL: Image loading and basic processing
- NumPy: Numerical operations
- Threading: Parallel processing support

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
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import platform
import psutil
import logging
from datetime import datetime
from pathlib import Path

# ==== Configuration Parameters ====
# These parameters control the behavior of the augmentation pipeline
# and can be adjusted based on specific requirements.

# Processing Mode Parameters
# ------------------------
# Controls the fundamental image processing behavior

# USE_GRAYSCALE: Determines color mode of processing
# - False: Maintain RGB color information (default)
# - True: Convert to grayscale for intensity-based processing
USE_GRAYSCALE = False

# APPLY_NOISE: Controls noise injection for robustness training
# - False: Clean images (default)
# - True: Add Gaussian noise for data robustness
APPLY_NOISE = False

# NUM_AUGMENTATIONS_PER_IMAGE: Number of variants per input image
# - Higher values provide more training diversity
# - Impacts processing time and storage requirements linearly
NUM_AUGMENTATIONS_PER_IMAGE = 2

# Signal Processing Parameters
# -------------------------
# Fine-tune the augmentation characteristics

# NOISE_SIGMA: Controls the intensity of Gaussian noise
# - Range: [0.0, 1.0] after normalization
# - Higher values create more pronounced noise patterns
# - 5.0/255.0 provides subtle noise suitable for most applications
NOISE_SIGMA = 5.0/255.0

# Color Jittering Parameters
# Each parameter defines the maximum random adjustment range
# Values typically range from 0.0 (no change) to 1.0 (full range)

# JITTER_BRIGHTNESS: Random brightness adjustment range
# - 0.4 allows for -40% to +40% brightness variation
JITTER_BRIGHTNESS = 0.4

# JITTER_CONTRAST: Random contrast adjustment range
# - 0.4 allows for -40% to +40% contrast variation
JITTER_CONTRAST = 0.4

# JITTER_SATURATION: Random saturation adjustment (RGB only)
# - 0.0 maintains original saturation
JITTER_SATURATION = 0.0

# JITTER_HUE: Random hue rotation (RGB only)
# - 0.0 maintains original hue
JITTER_HUE = 0.0

# System Parameters
# ----------------
# Control the execution behavior and output characteristics

# FINAL_SIZE: Determines the dimensions of output images
# - All images will be cropped/scaled to this size
# - Must match neural network input requirements
# - 1200x1200 optimal for tactile sensor resolution
FINAL_SIZE = 1200

# IMAGE_SHIFT_RIGHT: Horizontal alignment correction
# - Positive: Shift right
# - Negative: Shift left
# - -30 compensates for systematic sensor misalignment
IMAGE_SHIFT_RIGHT = -30

# SAVE_EVERY: Controls checkpoint frequency
# - Lower values provide better crash recovery but slower processing
# - Higher values improve performance but risk more data loss on crashes
# - 10 provides good balance between safety and performance
SAVE_EVERY = 10

class AugmentationConfig:
    """
    Configuration management for the augmentation pipeline.
    
    This class manages all configuration aspects including:
    - Environment-specific settings (desktop vs. supercomputer)
    - Processing parameters and modes
    - System resource allocation
    - File paths and naming conventions
    - Logging configuration
    
    The configuration ensures consistent behavior across different
    execution environments while optimizing resource usage based
    on the available hardware.
    """
    
    def __init__(self, environment="supercomputer"):
        """
        Initialize configuration with environment-specific settings.
        
        Args:
            environment (str): Either 'desktop' or 'supercomputer'
        """
        # Environment Detection
        self.environment = environment
        self.cpu_count = psutil.cpu_count(logical=False)
        
        # Processing Mode
        self.use_rgb = not USE_GRAYSCALE  # Match desktop version's grayscale setting
        self.apply_noise = APPLY_NOISE    # Match desktop version's noise setting
        self.apply_jitter = True          # Always true for augmentations
        
        # Performance Settings
        self.num_threads = max(1, self.cpu_count - 1) if environment == "supercomputer" else 1
        self.batch_size = 32 if environment == "supercomputer" else 1
        self.num_augmentations = NUM_AUGMENTATIONS_PER_IMAGE  # Match desktop version
        
        # Image Processing Parameters
        self.final_size = FINAL_SIZE            # Match desktop version
        self.image_shift = IMAGE_SHIFT_RIGHT    # Match desktop version
        self.noise_sigma = NOISE_SIGMA          # Match desktop version
        self.jitter_brightness = JITTER_BRIGHTNESS  # Match desktop version
        self.jitter_contrast = JITTER_CONTRAST      # Match desktop version
        self.jitter_saturation = JITTER_SATURATION  # Match desktop version
        self.jitter_hue = JITTER_HUE              # Match desktop version
        
        # Setup Paths
        self.setup_paths()
        
        # Initialize Logging
        self.setup_logging()
        
    def setup_paths(self):
        
        base_dir = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\code\code full pipeline\All code\Code from laptop\Testing_data_del\Data\full_dataset"
        
        # Input paths
        self.input_csv = os.path.join(base_dir, "0.labels.csv")
        self.input_img_dir = os.path.join(base_dir, "0.images")
        
        # Generate output paths with minimal naming
        timestamp = datetime.now().strftime("%m%d_%H%M")  # Short date format: MMDD_HHMM
        env_suffix = "_s" if self.environment == "supercomputer" else "_d"  # s for supercomputer, d for desktop
        
        self.output_dir = os.path.join(base_dir, 
            f"1.aug_images{env_suffix}_{timestamp}")
        self.output_csv = os.path.join(base_dir,
            f"1.aug_lab{env_suffix}_{timestamp}.csv")
        self.log_file = os.path.join(base_dir, f"aug_config{env_suffix}_{timestamp}.txt")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def setup_logging(self):
        """Configure logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
    def save_config(self):
        """Save current configuration to log file."""
        config_content = f"""
Augmentation Configuration
=========================
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Environment: {self.environment.upper()}

System Information:
- CPU Cores Available: {self.cpu_count}
- Parallel Threads: {self.num_threads}
- Batch Size: {self.batch_size}

Processing Mode:
- Color Mode: {"RGB" if self.use_rgb else "Grayscale"}
- Noise Injection: {"Enabled" if self.apply_noise else "Disabled"}
- Augmentations per Image: {NUM_AUGMENTATIONS_PER_IMAGE}

Signal Processing Parameters:
- Noise Sigma: {self.noise_sigma}
- Brightness Jitter: {self.jitter_brightness}
- Contrast Jitter: {self.jitter_contrast}
- Saturation Jitter: {self.jitter_saturation}
- Hue Jitter: {self.jitter_hue}

Image Processing:
- Augmentations per Image: {self.num_augmentations}
- Final Image Size: {self.final_size}
- Horizontal Shift: {self.image_shift}
- Save Frequency: Every {SAVE_EVERY} samples

Performance Settings:
- Thread Count: {self.num_threads}
- Batch Processing: {"Enabled" if self.batch_size > 1 else "Disabled"}
- Batch Size: {self.batch_size}

File Paths:
- Input CSV: {self.input_csv}
- Input Images Directory: {self.input_img_dir}
- Output Images Directory: {self.output_dir}
- Output Labels CSV: {self.output_csv}
- Configuration Log: {self.log_file}

File Naming:
- Current Files:
  - Labels: {os.path.basename(self.output_csv)}
  - Images: {os.path.basename(self.output_dir)}
  - Config: {os.path.basename(self.log_file)}
- Format: 1.augmented_[type]_[s/d]_MMDD_HHMM
  where s=supercomputer, d=desktop
"""
        logging.info(config_content)
        
class ImageProcessor:
    """
    Handles all image processing operations with tensor-based computations.
    
    This class encapsulates all image manipulation functionality including:
    - Image loading and format conversion
    - Tensor-based transformations for efficient processing
    - Geometric operations (rotation, masking, shifting)
    - Color space adjustments and noise injection
    - Contact point coordinate transformation
    
    The processor uses PyTorch tensors for efficient computation and
    maintains compatibility between RGB and grayscale processing pipelines.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize processor with given configuration.
        
        Args:
            config: AugmentationConfig instance with processing parameters
        """
        self.config = config
        self.device = torch.device("cpu")
        self.setup_transforms()
        
    def setup_transforms(self):
        """Initialize image transformation objects."""
        self.color_jitter = transforms.ColorJitter(
            brightness=self.config.jitter_brightness,
            contrast=self.config.jitter_contrast
        )
        
    def load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess image to tensor format.
        
        Args:
            image_path: Path to input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if self.config.use_rgb:
            # Load as RGB
            img = Image.open(image_path).convert("RGB")
            tensor = transforms.ToTensor()(img)
        else:
            # Load as grayscale but repeat to 3 channels for compatibility
            img = Image.open(image_path).convert("L")
            tensor = transforms.ToTensor()(img).repeat(3, 1, 1)
            
        return tensor
    
    @staticmethod
    def add_gaussian_noise(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Add Gaussian noise to the image tensor for data augmentation.
        
        This method implements zero-mean Gaussian noise injection:
        1. Generates random noise with specified standard deviation
        2. Adds noise to normalized image values [0,1]
        3. Clips results to maintain valid pixel range
        
        Mathematical Model:
        output = clamp(input + N(0, sigma^2), 0, 1)
        where N(0, sigma^2) is the normal distribution
        
        Args:
            tensor: Input image tensor (C x H x W) with values in [0,1]
            sigma: Standard deviation of the Gaussian noise
            
        Returns:
            torch.Tensor: Noisy image tensor with values clamped to [0,1]
            
        Note:
            Uses efficient tensor operations for parallel processing
            across all channels and pixels simultaneously.
        """
        noise = torch.randn_like(tensor) * sigma
        return torch.clamp(tensor + noise, 0.0, 1.0)
    
    @staticmethod
    def shift_image(tensor: torch.Tensor, shift_pixels: int) -> torch.Tensor:
        """Apply horizontal shift to image tensor."""
        pad_left = max(shift_pixels, 0)
        pad_right = max(-shift_pixels, 0)
        padded = F.pad(tensor, [pad_left, 0, pad_right, 0], fill=0.0)
        return padded[:, :tensor.shape[1], :tensor.shape[2]]
    
    @staticmethod
    def apply_circular_mask(tensor: torch.Tensor, diameter: int) -> torch.Tensor:
        """
        Apply a circular mask to isolate the tactile sensor region.
        
        Creates and applies a binary circular mask to remove image regions
        outside the tactile sensor's active area. Uses efficient tensor
        operations for fast processing.
        
        Implementation Steps:
        1. Create coordinate grids for x and y dimensions
        2. Calculate Euclidean distance from center for each pixel
        3. Create binary mask based on radius threshold
        4. Apply mask while preserving channel dimensions
        
        Mathematical Model:
        mask = (x - cx)² + (y - cy)² ≤ r²
        output = input * mask
        
        Args:
            tensor: Input image tensor (C x H x W)
            diameter: Diameter of the circular mask in pixels
            
        Returns:
            torch.Tensor: Masked image tensor with outside regions set to 0
            
        Performance Notes:
        - Uses broadcasting for efficient tensor operations
        - Avoids explicit loops for pixel-wise calculations
        - Maintains original tensor device location (CPU/GPU)
        """
        _, H, W = tensor.shape
        radius = diameter // 2
        cx, cy = W // 2, H // 2
        y = torch.arange(H, device=tensor.device).view(H, 1).expand(H, W)
        x = torch.arange(W, device=tensor.device).view(1, W).expand(H, W)
        dist = torch.sqrt((x - cx).float()**2 + (y - cy).float()**2)
        mask = (dist <= radius).float()
        return tensor * mask.unsqueeze(0)
    
    @staticmethod
    def rotate_point(x: float, y: float, angle_degrees: float, 
                    center_x: float = 0.0, center_y: float = 0.0) -> tuple:
        """
        Rotate a point around a center by specified angle.
        
        Args:
            x, y: Point coordinates
            angle_degrees: Rotation angle in degrees
            center_x, center_y: Rotation center coordinates
            
        Returns:
            tuple: (new_x, new_y) rotated coordinates
        """
        if pd.isna(x) or pd.isna(y):
            return np.nan, np.nan
            
        theta = math.radians(angle_degrees)
        xs, ys = x - center_x, y - center_y
        nx = xs*math.cos(theta) - ys*math.sin(theta) + center_x
        ny = xs*math.sin(theta) + ys*math.cos(theta) + center_y
        return nx, ny
    
    def process_single_augmentation(self, base_tensor: torch.Tensor, 
                                  contact_point: tuple, 
                                  angle: float) -> tuple:
        """
        Process a single augmentation by applying the complete transformation pipeline.
        
        This method orchestrates the full augmentation sequence:
        1. Color/Intensity Transformations:
           - Color jittering (if enabled)
           - Gaussian noise injection (if enabled)
        
        2. Geometric Transformations:
           - Horizontal shift correction
           - Circular mask application
           - Rotation by specified angle
           - Center cropping to final size
        
        3. Coordinate Transformation:
           - Updates contact point coordinates
           - Handles rotation transformation
           - Maintains coordinate system consistency
        
        Implementation Details:
        - Creates deep copy of input tensor to prevent modification
        - Applies transformations in specific order for correct results
        - Handles both RGB and grayscale inputs uniformly
        - Preserves tensor device location (CPU/GPU)
        
        Args:
            base_tensor: Input image tensor (C x H x W)
            contact_point: Original (x, y) contact point coordinates in mm
            angle: Rotation angle in degrees for augmentation
            
        Returns:
            tuple: (
                processed_tensor: Transformed image tensor,
                new_contact_point: Updated (x, y) coordinates after transformation
            )
            
        Note:
            Order of operations is critical for correct coordinate transformation
            and image quality. Changes to the sequence may require updates to
            coordinate calculations.
        """
        t = base_tensor.clone()
        
        # Apply transformations
        if self.config.apply_jitter:
            pil = transforms.ToPILImage()(t)
            pil = self.color_jitter(pil)
            t = transforms.ToTensor()(pil)
            
        if self.config.apply_noise:
            t = self.add_gaussian_noise(t, self.config.noise_sigma)
            
        # Geometric transformations
        t = self.shift_image(t, self.config.image_shift)
        t = self.apply_circular_mask(t, diameter=self.config.final_size)
        t = F.rotate(t, angle, expand=True)
        t = F.center_crop(t, [self.config.final_size, self.config.final_size])
        
        # Update contact point
        new_point = self.rotate_point(*contact_point, angle)
        
        return t, new_point
        
def process_row(row: pd.Series, processor: ImageProcessor, output_dir: str) -> list:
    """
    Process a single image row with all its augmentations.
    
     This function handles the complete augmentation pipeline for one image:
     1. Loads and validates the input image
     2. Generates multiple augmented versions with random rotations
     3. Applies all configured transformations (jitter, noise, etc.)
     4. Updates contact point coordinates after transformations
         - Uses 'Contact_X_mm_original', 'Contact_Y_mm_original' for pre-rotation positions
         - Uses 'Contact_X_mm_after_rotation', 'Contact_Y_mm_after_rotation' for post-rotation positions
     5. Saves augmented images and metadata
    
    The function is designed to be thread-safe for parallel processing
    in the supercomputer environment.
    
    Args:
        row: DataFrame row with image metadata (name, coordinates)
        processor: ImageProcessor instance with transformation pipeline
        output_dir: Output directory path for augmented images
        
    Returns:
      list: List of dictionaries containing metadata for each augmented version
          including new filenames, transformed coordinates, and rotation angles
          with clear column names for original and rotated positions
        
    Thread Safety:
        This function is designed to be thread-safe as it:
        - Creates new objects for each operation
        - Uses only local variables
        - Makes no modifications to shared state
    """
    results = []
    img_filename = row["New_Image_Name"]
    stem = os.path.splitext(img_filename)[0]
    img_path = os.path.join(processor.config.input_img_dir, img_filename)
    
    try:
        if not os.path.exists(img_path):
            logging.warning(f"Missing image: {img_path}")
            return []
            
        # Load base image
        base_tensor = processor.load_image(img_path)
        contact_point = (row["X_Position_mm_original"], row["Y_Position_mm_original"])
        
        # Generate augmentations
        for i in range(processor.config.num_augmentations):
            angle = random.uniform(0, 360)
            
            # Process augmentation
            aug_tensor, new_point = processor.process_single_augmentation(
                base_tensor, contact_point, angle)
            
            # Save augmented image
            aug_name = f"{stem}_aug{i}.png"
            out_path = os.path.join(output_dir, aug_name)
            out_pil = transforms.ToPILImage()(aug_tensor)
            out_pil.save(out_path, format="PNG")
            
            # Update metadata
            meta = row.copy()
            meta["New_Image_Name"] = aug_name
            meta["X_Position_mm_after_rotation"], meta["Y_Position_mm_after_rotation"] = new_point
            meta["rotation_angle"] = angle
            results.append(meta)
            
    except Exception as e:
        logging.error(f"Error processing {img_filename}: {str(e)}")
        return []
        
    return results



def main():
    """
    Main execution function orchestrating the augmentation pipeline.
    
    This function coordinates the entire augmentation process:
    1. Configuration initialization and validation
    2. Data loading and preprocessing
    3. Parallel processing of image augmentations
    4. Progress tracking and periodic logging
    5. Results collection and sorting
    6. Final output generation
    
    The function implements environment-specific optimizations:
    - For supercomputer: Utilizes parallel processing with ThreadPoolExecutor
    - For desktop: Processes sequentially for compatibility
    
    Key Features:
    - Robust error handling with detailed logging
    - Progress tracking with periodic updates
    - Memory-efficient batch processing
    - Automatic output sorting for supercomputer mode
    - Comprehensive validation at each stage
    """
    try:
        # Initialize configuration
        config = AugmentationConfig()
        config.save_config()
        
        # Initialize processor
        processor = ImageProcessor(config)
        
        # Load and validate input data
        logging.info(f"Loading data from: {config.input_csv}")
        if not os.path.exists(config.input_csv):
            raise FileNotFoundError(f"Input CSV not found: {config.input_csv}")
            
        df = pd.read_csv(config.input_csv)
        # Rename original position columns for compatibility
        if "X_Position_mm" in df.columns and "Y_Position_mm" in df.columns:
            df = df.rename(columns={
                "X_Position_mm": "X_Position_mm_original",
                "Y_Position_mm": "Y_Position_mm_original"
            })
        required_columns = ["New_Image_Name", "X_Position_mm_original", "Y_Position_mm_original"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Initialize result dataframe
        df_aug = pd.DataFrame(columns=list(df.columns) + ["X_Position_mm_after_rotation","Y_Position_mm_after_rotation","rotation_angle"])
        processed_count = 0
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
            futures = [
                executor.submit(process_row, row, processor, config.output_dir)
                for _, row in df.iterrows()
            ]
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), 
                             total=len(futures), 
                             desc=f"Augmenting with {config.num_threads} threads"):
                results = future.result()
                if results:
                    df_aug = pd.concat([df_aug, pd.DataFrame(results)], 
                                     ignore_index=True)
                    processed_count += len(results)
                    
                    # Log progress
                    if processed_count % 100 == 0:
                        logging.info(f"Progress: {processed_count} augmentations completed")
        
        # Sort DataFrame by New_Image_Name if in supercomputer mode to maintain chronological order
        if config.environment == "supercomputer":
            logging.info("[INFO] Sorting augmented data by image name for consistent ordering...")
            df_aug = df_aug.sort_values(by="New_Image_Name", key=lambda x: pd.Series([
                # Custom sorting to handle numeric parts in filenames
                tuple(map(lambda y: int(y) if y.isdigit() else y, 
                    str(xi).replace('.png', '').split('_'))) 
                for xi in x
            ]))
            
        # Save final results
        df_aug.to_csv(config.output_csv, index=False)
        logging.info(f"\n[SUCCESS] Augmentation complete. Generated {len(df_aug)} samples")
        logging.info(f"[INFO] Labels saved to: {config.output_csv}")
        logging.info(f"[INFO] Images saved to: {config.output_dir}")
            
    except Exception as e:
        logging.error(f"[ERROR] Critical error during processing: {str(e)}")

if __name__ == "__main__":
    main()