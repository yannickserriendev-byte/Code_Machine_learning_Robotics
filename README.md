# 🧠 Real-Time Multi-Task Tactile Sensing Pipeline

**End-to-end workflow for tactile sensing research using photoelastic fringe patterns and deep learning.**

## 🚀 Quick Start (3 Steps)

1. **Acquire Data**: Run acquisition pipeline to collect sensor and image data
2. **Prepare & Augment**: Preprocess and augment datasets for ML training
3. **Train & Infer**: Train models, run inference, and analyze results


## 🧩 Simple Pipeline: File Execution Sequence
Run these files in order for a complete workflow:

1. `acquisition/main_acquisition_pipeline.py`  
	*Orchestrates the entire tactile sensing experiment: interactive setup, synchronized force sensor and camera data collection, robust file organization, dropped frame diagnostics, force-to-image mapping, optional filtering, spatial labeling, and visualization. Produces a complete, ML-ready dataset from hardware.*
2. `preprocessing/combine_full_dataset.py`  
	*Combines multiple trial folders into a unified dataset. Merges images and sensor measurements, matches force and position data, renames and renumbers files, and creates a single metadata CSV for downstream ML. Handles missing data and enforces consistent structure.*
3. `augmentation/augment_images_desktop_and_super_computer.py`  
	*Generates diverse augmented image-label pairs for ML. Applies geometric and color transformations, noise injection, and masking. Tracks contact points through transformations, supports both desktop and supercomputer batch processing, and outputs robust, timestamped CSVs.*
4. `postprocessing/postprocess_labels.py`  
	*Cleans and standardizes augmented label CSVs: converts forces to absolute values, sets shape/position to None/NaN for no-contact frames, renames columns, sorts by image/augmentation, remaps shapes to integer classes, and saves a final, ML-compatible CSV.*
5. `training/train_multitask_model.py`  
	*Trains and evaluates multi-task deep learning models (ResNet18 or custom CNN) for force, class, and contact point prediction. Loads config and dataset, splits data, runs training loop with validation, checkpoints, computes metrics, and saves results/plots.*
6. `inference/main.py`  
	*Runs real-time or batch inference using trained models on new or live camera data. Dynamically loads model and dataset, preprocesses images, predicts force/class/contact point, and outputs results. Supports both grayscale/RGB and CPU/GPU modes.*

---

## 📋 Requirements

**Hardware:**
- NI-DAQ device with 6-axis force sensor (for acquisition)
- Basler camera for tactile imaging (for acquisition)

**Software:**
```bash
pip install torch torchvision pandas matplotlib scikit-learn seaborn pillow nidaqmx pypylon opencv-python scipy numpy
```

## 🔄 Complete Workflow

1. **📋 Data Acquisition** - Collect synchronized force and image data
2. **🧹 Preprocessing** - Combine and clean raw datasets
3. **🧬 Augmentation** - Expand dataset with robust image/label transformations
4. **📊 Postprocessing** - Analyze, visualize, and refine model outputs
5. **🧠 Model Training** - Multi-task learning for force, class, and contact point
6. **🔮 Inference** - Predict on new data or live camera input


## 📁 Output Structure

```
Project/
├── acquisition/
│   └── Trial_XXX/
│       ├── measurement_setup_data.csv
│       ├── Images/
│       ├── force.csv
│       ├── *_timestamps.csv
│       └── Results/
│           ├── unfiltered_final_frame_force_mapping.csv
│           ├── filtered_final_frame_force_mapping.csv
│           ├── positions_indentation.csv
│           └── highlighted_forces_dual_axis.png
├── preprocessing/
│   └── full_dataset.csv
├── augmentation/
│   └── augmented_images/
│   └── augmented_labels.csv
├── training/
│   └── model_weights.pth
│   └── training_loss_plot.png
│   └── metrics_summary.json
├── inference/
│   └── predictions.csv
│   └── live_results/
├── postprocessing/
│   └── analysis_plots/
│   └── processed_labels.csv
```

## ⚙️ Key Configuration

**Edit these parameters in the relevant scripts:**

```python
# Acquisition
acquisition_time = 15
force_acquisition_frequency = 5000
acquisition_frequency_camera = 50
Max_Number_of_allowed_lost_frames = 55

# Preprocessing/Augmentation
input_data_path = "acquisition/Trial_XXX/Results/"
output_data_path = "preprocessing/full_dataset.csv"

# Training
MODEL_TYPE = "resnet18"  # or "owncnn"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

# Inference
model_path = "training/model_weights.pth"
input_images = "inference/new_images/"
```

## 🔧 Individual Modules

| Module/Folder      | Purpose                                 | When to Use Separately                |
|--------------------|-----------------------------------------|---------------------------------------|
| acquisition/       | Data collection from hardware           | New experiments, hardware changes     |
| preprocessing/     | Dataset combination and cleaning        | Custom dataset formats                |
| augmentation/      | Image/label augmentation                | Robustness testing, new transforms    |
| training/          | Model training and evaluation           | Model architecture changes            |
| inference/         | Run trained models on new data          | Live demo, batch prediction           |
| postprocessing/    | Output analysis and visualization       | Custom analysis, publication figures  |

## 🔧 Troubleshooting

**Common Issues:**
- **Hardware not detected**: Check connections and drivers
- **Frame loss**: Lower camera frequency or check system performance
- **Model training fails**: Check dataset paths and config
- **Inference errors**: Verify model weights and input formats

**Quality Indicators:**
- Frame loss < threshold (see acquisition README)
- Consistent force/image synchronization
- Model metrics meet research targets

---

💡 **Pro Tip**: Start with small datasets and short acquisition times to validate your setup before scaling up.

For detailed parameter explanations and advanced configuration, see inline documentation in each module's main script.
