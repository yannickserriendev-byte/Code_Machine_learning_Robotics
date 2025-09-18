# 🔮 Model Inference Pipeline

**Run trained models on new tactile sensor data, including live camera input and batch prediction.**

## 🚀 Quick Start (3 Steps)

1. **Prepare Model & Data**: Place trained model weights and new images in the input folder
2. **Configure**: Edit parameters in `main.py` (model type, device, paths)
3. **Run**: `python main.py` → Get predictions for force, class, and contact point

## 📋 Requirements

**Software:**
```bash
pip install torch torchvision pandas opencv-python pillow
```

## 🔄 Complete Workflow

1. **📥 Load Model & Data** - Read trained weights and new images
2. **🔄 Preprocessing** - Apply image transforms (grayscale/RGB)
3. **🧠 Inference** - Run model to predict outputs
4. **📊 Output Handling** - Print and save predictions
5. **📷 Live Camera** - Optionally run inference on live camera feed

## 📁 Output Structure

```
inference/
├── predictions.csv            # Batch predictions on new data
├── live_results/              # Live camera inference outputs
```

## ⚙️ Key Configuration

**Edit these parameters in `main.py`:**

```python
MODEL_TYPE = "resnet18"  # or "owncnn"
DEVICE = "cuda"  # or "cpu"
model_path = "../training/model_weights.pth"
input_images = "new_images/"
color_mode = "grayscale"  # or "rgb"
```

## 🔧 Individual Modules

| Module              | Purpose                        | When to Use Separately           |
|---------------------|--------------------------------|----------------------------------|
| main.py             | Unified inference script       | Live demo, batch prediction      |
| inference_dataset.py| Load/format inference data     | Custom input formats             |
| model_ResNet18.py   | ResNet18 inference model       | Use standard architecture        |
| model_own_CNN.py    | Custom CNN inference model     | Use custom architecture          |

## 🔧 Troubleshooting

**Common Issues:**
- **Model not loaded**: Check model path and device
- **Image format errors**: Ensure input images match expected format
- **Live camera fails**: Check camera drivers and permissions

**Quality Indicators:**
- Predictions match expected ranges
- No missing or failed outputs

---

💡 **Pro Tip**: Use live inference for real-time demos and batch mode for large-scale evaluation.
