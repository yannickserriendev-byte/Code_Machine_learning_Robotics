"""
General multitask model training script for tactile sensing.

- Select model type ('resnet18' or 'simplecnn') via MODEL_TYPE variable below.
- Loads images and labels, splits into train/val/test sets.
- Trains multitask model (force regression, shape classification, contact point regression).
- Usage: Adjust MODEL_TYPE, paths, and parameters as needed.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from Code_Machine_learning_Robotics.utils.dataset import IsochromaticDataset
from Code_Machine_learning_Robotics.models.resnet18_multitask import IsoNet
from Code_Machine_learning_Robotics.models.simplecnn_multitask import SimpleCNN

# ==== Hyperparameters ====
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Paths ====
base_dir = "<SET TO YOUR DATASET ROOT>"
image_dir = os.path.join(base_dir, "images")
labels_path = os.path.join(base_dir, "labels.csv")

# ==== Model output folder ====
model_dir = os.path.join(base_dir, "model_output")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "isonet_final.pth")
eval_output_dir = os.path.join(model_dir, "evaluation")
os.makedirs(eval_output_dir, exist_ok=True)
split_path = os.path.join(model_dir, "split_indices.json")

# ==== Transforms ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ==== Load Dataset ====
dataset = IsochromaticDataset(images_folder=image_dir, labels_csv=labels_path, transform=transform)
num_classes = len(dataset.get_shape_labels())

dataset_size = len(dataset)
val_size = int(VAL_SPLIT * dataset_size)
test_size = int(TEST_SPLIT * dataset_size)
train_size = dataset_size - val_size - test_size

if os.path.exists(split_path):
    with open(split_path, "r") as f:
        split_indices = json.load(f)
    train_set = Subset(dataset, split_indices["train"])
    val_set = Subset(dataset, split_indices["val"])
    test_set = Subset(dataset, split_indices["test"])
    print("✅ Loaded saved split indices.")
else:
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    with open(split_path, "w") as f:
        json.dump({
            "train": train_set.indices,
            "val": val_set.indices,
            "test": test_set.indices
        }, f)
    print("📁 New split created and saved.")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)


# ==== Model Selection ====
# Set MODEL_TYPE to 'resnet18' or 'simplecnn' to choose architecture
MODEL_TYPE = 'resnet18'  # or 'simplecnn'
if MODEL_TYPE == 'resnet18':
    model = IsoNet(num_shape_classes=num_classes).to(DEVICE)
elif MODEL_TYPE == 'simplecnn':
    model = SimpleCNN(num_shape_classes=num_classes).to(DEVICE)
else:
    raise ValueError('Unknown MODEL_TYPE')
force_loss_fn = nn.MSELoss()
class_loss_fn = nn.CrossEntropyLoss()
point_loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== Training ====
train_losses, val_losses = [], []
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for images, forces, shape_class, contact_point in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, forces, shape_class, contact_point = images.to(DEVICE), forces.to(DEVICE), shape_class.to(DEVICE), contact_point.to(DEVICE)
        optimizer.zero_grad()
        pred_forces, pred_class, pred_point = model(images)
        loss = (
            force_loss_fn(pred_forces, forces)
            + class_loss_fn(pred_class, shape_class)
            + point_loss_fn(pred_point, contact_point)
        )
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train = train_loss / len(train_loader)
    train_losses.append(avg_train)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, forces, shape_class, contact_point in val_loader:
            images, forces, shape_class, contact_point = images.to(DEVICE), forces.to(DEVICE), shape_class.to(DEVICE), contact_point.to(DEVICE)
            pred_forces, pred_class, pred_point = model(images)
            val_loss += (
                force_loss_fn(pred_forces, forces)
                + class_loss_fn(pred_class, shape_class)
                + point_loss_fn(pred_point, contact_point)
            ).item()
    avg_val = val_loss / len(val_loader)
    val_losses.append(avg_val)
    print(f"📉 Epoch {epoch+1} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

# Save final model
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to: {model_path}")

# ==== Final Test Evaluation ====
print("\n🧪 Final evaluation on test set...")
model.eval()
test_force_loss, test_class_loss, test_point_loss = 0, 0, 0
criterion_ce = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()
with torch.no_grad():
    for images, forces, shape_class, contact_point in tqdm(test_loader, desc="Testing"):
        images, forces, shape_class, contact_point = images.to(DEVICE), forces.to(DEVICE), shape_class.to(DEVICE), contact_point.to(DEVICE)
        pred_forces, pred_class, pred_point = model(images)
        test_force_loss += criterion_mse(pred_forces, forces).item() * images.size(0)
        test_class_loss += criterion_ce(pred_class, shape_class).item() * images.size(0)
        test_point_loss += criterion_mse(pred_point, contact_point).item() * images.size(0)
n = len(test_set)
print(f"\n📊 Final TEST metrics over {n} samples:")
print(f"  🔧 Force MSE    : {test_force_loss / n:.4f}")
print(f"  🎯 Class CE Loss: {test_class_loss / n:.4f}")
print(f"  📍 Point MSE    : {test_point_loss / n:.4f}")
