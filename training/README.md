# 🧠 Multi-Task Model Training Pipeline

**Train deep learning models to predict force, class, and contact point from tactile images.**

## 🚀 Quick Start (3 Steps)

1. **Prepare Data**: Place augmented images and labels in the input folder
2. **Configure**: Edit hyperparameters and paths in `train_multitask_model.py`
3. **Run**: `python train_multitask_model.py` → Get trained model and metrics

## 📋 Requirements

**Software:**
```bash
pip install torch torchvision pandas matplotlib scikit-learn seaborn pillow
```

## 🔄 Complete Workflow

1. **📥 Load Data** - Read augmented images and labels
2. **🧠 Model Selection** - Choose architecture (ResNet18 or custom CNN)
3. **🎯 Training Loop** - Train model with multi-task heads
4. **💾 Checkpointing** - Save model weights and training progress
5. **📊 Evaluation** - Compute metrics and visualize results

## 📁 Output Structure

```
training/
├── model_weights.pth           # Trained model weights
├── training_loss_plot.png      # Training/validation loss curves
├── metrics_summary.json        # Evaluation metrics
```

## ⚙️ Key Configuration

**Edit these parameters in `train_multitask_model.py`:**

```python
MODEL_TYPE = "resnet18"  # or "owncnn"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
input_images = "../augmentation/augmented_images/"
input_labels = "../augmentation/augmented_labels.csv"
```

## 🔧 Individual Modules

| Module                | Purpose                        | When to Use Separately           |
|-----------------------|--------------------------------|----------------------------------|
| train_multitask_model.py | Train/evaluate model         | Model architecture changes       |
| data_preparation.py      | Load/format datasets         | Custom dataset formats           |
| ResNet18_multitask.py    | ResNet18 model definition    | Use standard architecture        |
| SimpleCNN_multitask.py   | Custom CNN model definition  | Use custom architecture          |

## 🔧 Troubleshooting

**Common Issues:**
- **Training fails**: Check input paths and data formats
- **Low accuracy**: Tune hyperparameters or try more augmentations
- **Checkpoint not saved**: Check disk space and permissions

**Quality Indicators:**
- Loss curves show convergence
- Metrics meet research targets

---

💡 **Pro Tip**: Use checkpoints to resume training and experiment with different architectures for best results.
