# Master Thesis Tactile Sensing Pipeline: In-Depth Usage Guide

This guide explains how to run the full tactile sensing pipeline, the order of execution, and the purpose of each code module. It is designed for new users and thesis readers to understand, reproduce, and extend the workflow.

---

## 1. Overview

The pipeline consists of the following stages:
- Acquisition: Collect raw images and force/position data from experiments.
- Preprocessing: Merge, clean, and relabel data for ML use.
- Augmentation: Generate augmented images and labels for robust training.
- Model Definition: Specify neural network architectures for multitask learning.
- Training: Train models on augmented data.
- Inference: Evaluate trained models on test data.
- Utilities: Helper scripts for data management and cleaning.
- Configuration: Cluster/job scripts and config files.
- Attic: Legacy/near-duplicate code for reference.

---

## 2. Folder Structure

- `acquisition/` — Scripts for raw data collection and mapping.
- `preprocessing/` — Scripts for merging, cleaning, and relabeling datasets.
- `augmentation/` — Scripts for image augmentation (Sensor 1/2, RGB, grayscale, noise, etc.).
- `models/` — Model definition files (ResNet18, SimpleCNN, multitask heads).
- `training/` — Training scripts for multitask models.
- `inference/` — Inference/evaluation scripts for trained models.
- `utils/` — Utility/helper scripts (CSV cleaning, dataset combining).
- `configs/` — Configuration files (e.g., SLURM job scripts).
- `.attic/` — Legacy/near-duplicate code (not used in main pipeline).

---

## 3. Step-by-Step Pipeline Execution

### Step 1: Acquisition
- **Purpose:** Collect raw images and force/position data from experiments.
- **Scripts:**
  - `_00_measurement_setup.py` — Setup and document measurement hardware.
  - `_01_acquisition.py` — Acquire images and sensor data.
  - `_02_saving_data.py` — Save raw images and CSVs.
  - `_03_analysis.py` — Analyze raw data for quality.
  - `_04_force_to_image_mapping.py` — Map force/position to image numbers.
  - `_05_low_pass_filter.py` — Filter sensor signals.
  - `_06_plotting.py` — Visualize acquisition results.
  - `_07_video_creation.py` — Create videos from image sequences.
- **Order:** Run scripts in numerical order as your experiment progresses.
- **Output:** Raw images (`0.images/`), force/position CSVs, documentation.

### Step 2: Preprocessing
- **Purpose:** Merge, clean, and relabel raw data for ML use.
- **Scripts:**
  - `combine_full_dataset.py` — Merge images and CSVs from all trials into a unified dataset.
  - `mark_invalid_xy_as_nan.py` — Set invalid (X, Y) positions to NaN.
  - `relabel_shape_classes.py` — Remap shape labels to integer classes.
  - `scale_force_and_torque.py` — Normalize force/torque values.
  - `relabel_xy_mistake_in_y_value.py` — Correct Y value mistakes in labels.
  - `relab_shape_class_correction.py` — Final correction of shape classes.
- **Order:**
  1. Run `combine_full_dataset.py` to create `0.images/` and `0.labels.csv`.
  2. Run cleaning/correction scripts in order as needed for your dataset.
- **Output:** Cleaned images and labels ready for augmentation.

### Step 3: Augmentation
- **Purpose:** Generate augmented images and labels for robust model training.
- **Scripts:**
  - `augment_images_sensor1.py` — Augment Sensor 1 images (RGB, configurable).
  - `augment_images_sensor2.py` — Augment Sensor 2 images (RGB, configurable).
  - `augment_images_sensor1_grayscale_noise.py` — Augment Sensor 1 images (grayscale + noise).
- **Order:**
  1. Choose the script matching your sensor and augmentation type.
  2. Configure parameters (input/output paths, augmentations per image, etc.).
  3. Run the script to generate augmented images and labels.
- **Output:** Augmented images and labels CSVs for model training.

### Step 4: Model Definition
- **Purpose:** Specify neural network architectures for multitask learning.
- **Files:**
  - `resnet18_multitask.py` — ResNet18 backbone multitask model.
  - `simplecnn_multitask.py` — SimpleCNN multitask model.
- **Order:** Select the model file matching your experiment. Import into training scripts.
- **Output:** Model classes for use in training/inference.

### Step 5: Training
- **Purpose:** Train multitask models on augmented data.
- **Scripts:**
  - `train_resnet18_multitask.py` — Train ResNet18 multitask model.
- **Order:**
  1. Configure paths, batch size, epochs, etc. in the script.
  2. Run the script to train the model. Use SLURM job script if on a cluster.
- **Output:** Trained model weights, training logs, split indices, plots.

### Step 6: Inference
- **Purpose:** Evaluate trained models on test data and report metrics.
- **Scripts:**
  - `inference_resnet18_multitask.py` — Run inference and evaluation on test set.
- **Order:**
  1. Configure paths and parameters in the script.
  2. Run the script to generate predictions, metrics, and confusion matrix.
- **Output:** Test predictions CSV, metrics, confusion matrix plot.

### Step 7: Utilities
- **Purpose:** Helper scripts for data management and cleaning.
- **Scripts:**
  - `clean_and_relabel_csv.py` — Clean and relabel augmented labels CSVs.
  - `combine_full_dataset.py` — Merge raw data into unified dataset.
- **Order:** Use as needed during preprocessing and data management.

### Step 8: Configuration
- **Purpose:** Cluster/job scripts and config files for reproducibility.
- **Files:**
  - `jobsubmit_train_example.sh` — SLURM job script for cluster training.
- **Order:** Edit and use as needed for your compute environment.

### Step 9: Attic
- **Purpose:** Archive legacy/near-duplicate code for reference.
- **Files:**
  - `.attic/` contains old versions of acquisition, model, training, and testing scripts.
- **Order:** Not used in main pipeline; for reference only.

---

## 4. General Workflow Example

1. **Acquire Data:** Run acquisition scripts during experiments to collect raw images and sensor data.
2. **Preprocess Data:** Merge and clean all raw data into a unified dataset using preprocessing scripts.
3. **Augment Data:** Generate augmented images and labels for robust model training.
4. **Define Model:** Select and configure the neural network architecture for multitask learning.
5. **Train Model:** Train the model using the training script, monitor loss and metrics.
6. **Run Inference:** Evaluate the trained model on test data, save predictions and metrics.
7. **Use Utilities:** Clean, relabel, or combine data as needed using utility scripts.
8. **Configure Jobs:** Use SLURM/config files for cluster training if needed.
9. **Reference Attic:** Consult legacy code for historical context or troubleshooting.

---

## 5. Tips & Best Practices
- Always check and set input/output paths in each script before running.
- Use the provided docstrings and inline comments for configuration guidance.
- Run scripts in the recommended order for reproducible results.
- Archive legacy code in `.attic/` to keep the main pipeline clean.
- For cluster training, use the SLURM job script and adjust resources as needed.
- Validate outputs at each stage before proceeding to the next.

---

## 6. Troubleshooting
- **Missing Packages:** Install required Python packages (numpy, pandas, torch, torchvision, tqdm, matplotlib, seaborn, scikit-learn, PIL).
- **Path Errors:** Double-check all input/output paths and folder names.
- **Legacy Code:** If unsure about a script, check if a newer version exists in the main folders.
- **Data Issues:** Use utility scripts to clean and relabel data as needed.

---

## 7. Extending the Pipeline
- Add new augmentation modes by extending scripts in `augmentation/`.
- Implement new model architectures in `models/` and update training/inference scripts.
- Add new utility scripts for data management in `utils/`.
- Document all changes for reproducibility and thesis alignment.

---

## 8. Contact & Support
For questions, troubleshooting, or further development, consult the inline comments and docstrings in each script, or contact the thesis author.

---

**End of Guide**
