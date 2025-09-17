# Tactile Sensing Data Processing Pipeline

This folder contains a complete data processing pipeline for tactile sensing data, organized in a clear and modular structure.

## Pipeline Overview

```
Code_Machine_learning_Robotics/
├── preprocessing/               # Data combination and preparation
│   └── combine_data.py         # Combines multiple trial datasets
├── data/                       # Data storage
│   ├── combined_dataset/       # Output from preprocessing
│   │   ├── images/            # Combined raw images
│   │   └── labels.csv         # Combined metadata
│   └── augmented_dataset/      # Output from augmentation
│       ├── images_*/          # Augmented images (with mode suffixes)
│       └── labels_*.csv       # Augmented labels (with mode suffixes)
└── augmentation/               # Data augmentation scripts
    └── augment_images_sensor1_hpc.py   # HPC-optimized augmentation

```

## Data Flow

1. **Data Combination** (`preprocessing/combine_data.py`)
   - Input: Raw trial data from multiple experiments
   - Process: Combines images and labels from all trials
   - Output: Standardized dataset in `data/combined_dataset/`

2. **Data Augmentation** (`augmentation/augment_images_sensor1_hpc.py`)
   - Input: Combined dataset from preprocessing step
   - Process: Applies configurable augmentations (rotation, noise, etc.)
   - Output: Augmented dataset in `data/augmented_dataset/`

## Usage

1. First, run the data combination script:
   ```bash
   python preprocessing/combine_data.py
   ```
   This will create a standardized dataset in the `data/combined_dataset/` directory.

2. Then, run the augmentation pipeline:
   ```bash
   python augmentation/augment_images_sensor1_hpc.py
   ```
   This will create augmented data in the `data/augmented_dataset/` directory.

## Configuration

- The base paths in both scripts can be modified to point to your specific data locations
- Augmentation parameters (number of augmentations, noise levels, etc.) can be configured in the augmentation script