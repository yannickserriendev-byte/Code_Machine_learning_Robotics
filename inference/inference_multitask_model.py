"""
General multitask model inference/evaluation script for tactile sensing.

- Select model type ('resnet18' or 'simplecnn') via MODEL_TYPE variable below.
- Loads trained model and test split, runs inference on test set.
- Computes force, class, and contact point metrics; saves predictions and confusion matrix.
- Usage: Adjust MODEL_TYPE, paths, and parameters as needed.
"""
import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from Code_Machine_learning_Robotics.utils.dataset import IsochromaticDataset
from Code_Machine_learning_Robotics.training.resnet18_multitask import IsoNet
from Code_Machine_learning_Robotics.training.simplecnn_multitask import SimpleCNN

# ==== Configuration ====
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Paths ====
base_dir = "<SET TO YOUR DATASET ROOT>"
model_dir = os.path.join(base_dir, "model_output")
image_dir = os.path.join(base_dir, "images")
labels_path = os.path.join(base_dir, "labels.csv")
model_path = os.path.join(model_dir, "isonet_final.pth")
split_path = os.path.join(model_dir, "split_indices.json")
output_dir = os.path.join(model_dir, "test_results")
os.makedirs(output_dir, exist_ok=True)

# ==== Transform ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ==== Load Dataset ====
dataset = IsochromaticDataset(images_folder=image_dir, labels_csv=labels_path, transform=transform)
num_classes = len(dataset.get_shape_labels())

# ==== Load Test Split ====
if not os.path.exists(split_path):
    raise FileNotFoundError(f"❌ split_indices.json not found at {split_path}")
with open(split_path, "r") as f:
    split_indices = json.load(f)
test_indices = split_indices["test"]
test_set = Subset(dataset, test_indices)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
print("✅ Loaded test split.")


# ==== Model Selection ====
# Set MODEL_TYPE to 'resnet18' or 'simplecnn' to choose architecture
MODEL_TYPE = 'resnet18'  # or 'simplecnn'
if MODEL_TYPE == 'resnet18':
    model = IsoNet(num_shape_classes=num_classes).to(DEVICE)
elif MODEL_TYPE == 'simplecnn':
    model = SimpleCNN(num_shape_classes=num_classes).to(DEVICE)
else:
    raise ValueError('Unknown MODEL_TYPE')
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print("✅ Loaded model from:", model_path)

# ==== Evaluation ====
force_loss_fn = nn.MSELoss()
class_loss_fn = nn.CrossEntropyLoss()
point_loss_fn = nn.MSELoss()
results = []
force_true_all, force_pred_all = [], []
point_true_all, point_pred_all = [], []
y_true_cls, y_pred_cls = [], []
for i, (images, forces, shape_classes, contact_points) in enumerate(tqdm(test_loader, desc="Testing")):
    images = images.to(DEVICE)
    forces = forces.to(DEVICE)
    shape_classes = shape_classes.to(DEVICE)
    contact_points = contact_points.to(DEVICE)
    with torch.no_grad():
        pred_forces, pred_classes, pred_points = model(images)
    force_true_all.append(forces.cpu().numpy())
    force_pred_all.append(pred_forces.cpu().numpy())
    point_true_all.append(contact_points.cpu().numpy())
    point_pred_all.append(pred_points.cpu().numpy())
    pred_classes_argmax = pred_classes.argmax(dim=1)
    y_true_cls += shape_classes.cpu().tolist()
    y_pred_cls += pred_classes_argmax.cpu().tolist()
    for j in range(images.size(0)):
        index = test_indices[i * BATCH_SIZE + j]
        img_name = dataset.data.iloc[index]["New_Image_Name"]
        results.append({
            "index": index,
            "image_file": img_name,
            **{f"true_force_{k}": v.item() for k, v in zip(['Fx','Fy','Fz','Mx','My','Mz','Ft'], forces[j])},
            **{f"pred_force_{k}": v.item() for k, v in zip(['Fx','Fy','Fz','Mx','My','Mz','Ft'], pred_forces[j])},
            "true_class": shape_classes[j].item(),
            "pred_class": pred_classes_argmax[j].item(),
            **{f"true_point_{k}": v.item() for k, v in zip(['x','y'], contact_points[j])},
            **{f"pred_point_{k}": v.item() for k, v in zip(['x','y'], pred_points[j])}
        })
# ==== Save CSV ====
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

# ==== Metrics ====
force_true_all = np.concatenate(force_true_all)
force_pred_all = np.concatenate(force_pred_all)
point_true_all = np.concatenate(point_true_all)
point_pred_all = np.concatenate(point_pred_all)
print("\n📊 Test Set Evaluation:")
print(f"  🔧 Force MSE: {force_loss_fn(torch.tensor(force_pred_all), torch.tensor(force_true_all)).item():.4f}")
print(f"  🎯 Class CE:  {class_loss_fn(torch.tensor(force_pred_all), torch.tensor(force_true_all)).item():.4f}")
print(f"  📍 Point MSE: {point_loss_fn(torch.tensor(point_pred_all), torch.tensor(point_true_all)).item():.4f}")
print("\n🎯 Classification Report:")
print(classification_report(y_true_cls, y_pred_cls, target_names=dataset.get_shape_labels()))
print(f"  ✅ Accuracy : {accuracy_score(y_true_cls, y_pred_cls):.4f}")
print(f"  ✅ Precision: {precision_score(y_true_cls, y_pred_cls, average='macro'):.4f}")
print(f"  ✅ Recall   : {recall_score(y_true_cls, y_pred_cls, average='macro'):.4f}")
print(f"  ✅ F1 Score : {f1_score(y_true_cls, y_pred_cls, average='macro'):.4f}")
# ==== Confusion Matrix ====
cm = confusion_matrix(y_true_cls, y_pred_cls)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=dataset.get_shape_labels(),
            yticklabels=dataset.get_shape_labels())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()
print(f"\n✅ All test results saved to: {output_dir}")
