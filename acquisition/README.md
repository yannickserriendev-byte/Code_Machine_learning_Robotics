# 🤖 Tactile Sensing Data Acquisition Pipeline

**Complete workflow from hardware setup to ML-ready datasets for real-time tactile sensing research.**

## 🚀 Quick Start (3 Steps)

1. **Hardware Setup**: Connect NI-DAQ force sensor + Basler camera
2. **Configure**: Edit parameters in `main_acquisition_pipeline.py` (paths, frequencies, calibration) (frequencies and calibration can be changes based on requirement/preference and calibration based on the force sensor being used)
3. **Run**: `python main_acquisition_pipeline.py` → Follow prompts → Click to start → Get ML dataset

## 📋 Requirements

**Hardware:**
- NI-DAQ device with 6-axis force sensor
- Basler camere for tactile imaging

**Software:**
```bash
pip install nidaqmx pypylon opencv-python pandas matplotlib scipy numpy
```

## 🔄 Complete Workflow

The pipeline automatically handles:

1. **📋 Interactive Setup** - Configure positions, indentors, measurement sequence
2. **🎯 Synchronized Acquisition** - Force sensor (5000Hz) + Camera (50-500Hz) # if increased too high then your computer might fail to save all images as it does not have a temporary storage and will use your computers CPU as a temporary storage device
3. **💾 Data Organization** - Structured file saving with timestamps
4. **🔍 Quality Control** - Frame loss analysis, timing diagnostics
5. **🔗 ML Dataset Creation** - Force-to-image mapping for training
6. **🎚️ Signal Processing** - Optional filtering for noise reduction
7. **📍 Spatial Labeling** - Position coordinates for each measurement
8. **📈 Visualization** - Verification plots and data analysis
9. **🎥 Video Generation** - Optional presentation videos

## 📁 Output Structure

Each trial creates:
```
Trial_XXX/
├── measurement_setup_data.csv      # Experimental parameters
├── Images/                         # Tactile images (TIFF)
├── force.csv                       # 6-axis force data
├── *_timestamps.csv               # Timing synchronization
└── Results/                       # ML-ready datasets
    ├── unfiltered_final_frame_force_mapping.csv  # Raw dataset
    ├── filtered_final_frame_force_mapping.csv    # Cleaned dataset
    ├── positions_indentation.csv                 # Spatial labels
    └── highlighted_forces_dual_axis.png          # Force visualization
```

## ⚙️ Key Configuration

**Edit these parameters in `main_acquisition_pipeline.py`:**

```python
# Data collection
acquisition_time = 15                    # Duration (seconds) # if increased too much your loss of images may increase
base_directory = "C:/your/data/path/"    # Save location

# Force sensor (NI-DAQ)
force_acquisition_frequency = 5000       # Sampling rate (Hz)
calibration_matrix = [[...]]            # From sensor .cal file

# Camera (Basler)
acquisition_frequency_camera = 50        # Frame rate (Hz) # if increased too much your loss of images may increase
Max_Number_of_allowed_lost_frames = 55   # Quality threshold # can be changed based on preference
```

## 🔧 Individual Modules

| Module | Purpose | When to Use Separately |
|--------|---------|----------------------|
| `_00_measurement_setup.py` | Interactive parameter configuration | Custom experimental setups |
| `_01_acquisition.py` | Hardware control classes | Integration with external systems |
| `_02_saving_data.py` | File organization functions | Custom data formats |
| `_03_analysis.py` | Quality diagnostics | Debugging acquisition issues |
| `_04_force_to_image_mapping.py` | ML dataset creation | Different temporal alignment needs |
| `_05_low_pass_filter.py` | Signal processing | Custom filtering requirements |
| `_06_plotting.py` | Data visualization | Custom analysis plots |
| `_07_video_creation.py` | Video generation | Presentation materials |


## 🔧 Troubleshooting

**Common Issues:**
- **Camera not detected**: Check Basler Pylon installation and USB/GigE connection
- **NI-DAQ errors**: Verify device name in NI MAX and/or ATIDAQFT.Net software 
- **High frame loss**: Reduce camera frequency or check system performance
- **Calibration errors**: Verify 6x6 matrix from sensor documentation

**Quality Indicators:**
- Frame loss < max threshold (typically <55 frames)
- Force sample timing consistency < 1% deviation
- Synchronized timestamps between force and camera data

---

💡 **Pro Tip**: Start with short acquisition times (5) to verify setup before longer experiments.

For detailed parameter explanations and advanced configuration, see inline documentation in `main_acquisition_pipeline.py`.
