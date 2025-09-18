# 🧬 Data Augmentation Pipeline

**Expand and diversify tactile sensor datasets for robust model training.**

## 🚀 Quick Start (3 Steps)

1. **Prepare Data**: Place cleaned dataset in the input folder
2. **Configure**: Edit augmentation parameters in `augment_images_desktop_and_super_computer.py`
3. **Run**: `python augment_images_desktop_and_super_computer.py` → Get augmented images and labels

## 📋 Requirements

**Software:**
```bash
pip install opencv-python numpy pandas
```

## 🔄 Complete Workflow

1. **📥 Load Dataset** - Read cleaned dataset from preprocessing
2. **🧬 Apply Augmentations** - Transform images (rotation, noise, color, etc.)
3. **🔗 Update Labels** - Adjust labels to match augmented images
4. **💾 Save Output** - Export augmented images and labels

## 📁 Output Structure

```
augmentation/
├── augmented_images/          # Augmented tactile images
├── augmented_labels.csv       # Updated labels for ML
```

## ⚙️ Key Configuration

**Edit these parameters in `augment_images_desktop_and_super_computer.py`:**

```python
input_dataset = "../preprocessing/full_dataset.csv"
output_folder = "augmented_images/"
num_augmentations = 3
color_mode = "grayscale"  # or "rgb"
noise_level = 0.05
```

## 🔧 Individual Modules

| Module                                 | Purpose                  | When to Use Separately         |
|----------------------------------------|--------------------------|-------------------------------|
| augment_images_desktop_and_super_computer.py | Image/label augmentation | Custom augmentation strategies |

## 🔧 Troubleshooting

**Common Issues:**
- **Images not saved**: Check output folder permissions
- **Label mismatch**: Ensure augmentation script updates labels correctly
- **Slow processing**: Reduce number of augmentations or image size

**Quality Indicators:**
- All augmented images have corresponding labels
- No duplicate or missing files

---

💡 **Pro Tip**: Experiment with different augmentation parameters to maximize model generalization.
