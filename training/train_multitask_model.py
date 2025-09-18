###############################################################
# train_multitask_model.py
# -------------------------------------------------------------
# Main training and evaluation script for multi-task tactile sensing models.
# - Loads configuration and dataset
# - Dynamically selects model architecture (ResNet18 or custom CNN)
# - Handles training loop, validation, checkpointing, and final evaluation
# - Reports metrics and saves plots/results for user analysis
# -------------------------------------------------------------

import os
import re
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
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import seaborn as sns

import sys
import os

# Add the script's folder to sys.path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import dataset class for loading tactile sensing data

from data_preparation import IsochromaticDataset

def get_model(model_type, num_shape_classes):
    """
    Dynamically import and instantiate the selected model architecture.
    Args:
        model_type (str): 'resnet18' or 'owncnn'.
        num_shape_classes (int): Number of shape classes for classification.
    Returns:
        torch.nn.Module: Instantiated model.
    """
    # Dynamically select the model architecture based on user config
    if model_type == "resnet18":
        from ResNet18_multitask import IsoNet as SelectedNet
    elif model_type == "owncnn":
        from SimpleCNN_multitask import IsoNet as SelectedNet
    else:
        raise ValueError("Unknown MODEL_TYPE: choose 'resnet18' or 'owncnn'")
    return SelectedNet(num_shape_classes=num_shape_classes)
    

def main():
    """
    Main entry point for training and evaluating multi-task tactile sensing models.
    Steps:
    1. Load user configuration and augmentation parameters
    2. Prepare dataset, split into train/val/test, and set up DataLoaders
    3. Dynamically select and initialize model architecture
    4. Resume from checkpoint if available, or start fresh
    5. Run training loop with validation and checkpointing
    6. Save final model and training/validation loss plots
    7. Evaluate on test set, report metrics, and save results
    """
    # ==== USER CONFIGURATION SECTION ====
    ENVIRONMENT = "desktop"  # "desktop" or "supercomputer"
    MODEL_TYPE = "owncnn"  # "resnet18" or "owncnn"
    BATCH_SIZE = 32
    NUM_EPOCHS = 2
    LEARNING_RATE = 1e-3
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    RANDOM_SEED = 42

    print(f"🔧 Hyperparameters: ENVIRONMENT={ENVIRONMENT}, MODEL_TYPE={MODEL_TYPE}, BATCH_SIZE={BATCH_SIZE}, NUM_EPOCHS={NUM_EPOCHS}, LEARNING_RATE={LEARNING_RATE}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {DEVICE}")
    
    # ==== Paths & Dynamic Naming ====
    # Path to aug_config file (update if needed)
    if ENVIRONMENT == "desktop":
        base_dir = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\code\code full pipeline\All code\Code from laptop\Testing_data_del\Data\full_dataset"
        aug_config_path = os.path.join(base_dir, "aug_config_d_0918_2005.txt")
        image_dir = os.path.join(base_dir, "1.aug_images_d_0918_2005")
        labels_path = os.path.join(base_dir, "2.aug_lab_postproc_d_0918_2005.csv")
    elif ENVIRONMENT == "supercomputer":
        base_dir = "/scratch/yserrrien/data aqcuisition/1/Data_Sensor_1/Crisp images/Data/full_dataset"
        aug_config_path = os.path.join(base_dir, "aug_config_s_0918_2005.txt")
        image_dir = os.path.join(base_dir, "1.aug_images_s_0918_2005")
        labels_path = os.path.join(base_dir, "2.aug_lab_postproc_s_0918_2005.csv")
    else:
        raise ValueError("Unknown ENVIRONMENT")

    # Parse aug_config for key parameters
    aug_params = {
        "color_mode": "unknown",
        "noise": "unknown",
        "aug_per_img": "unknown",
        "img_size": "unknown",
        "environment": ENVIRONMENT
    }
    try:
        with open(aug_config_path, "r") as f:
            for line in f:
                if "Color Mode:" in line:
                    aug_params["color_mode"] = line.split(":")[-1].strip()
                elif "Noise Injection:" in line:
                    aug_params["noise"] = line.split(":")[-1].strip()
                elif "Augmentations per Image:" in line:
                    aug_params["aug_per_img"] = line.split(":")[-1].strip()
                elif "Final Image Size:" in line:
                    aug_params["img_size"] = line.split(":")[-1].strip()
    except Exception as e:
        print(f"Warning: Could not read aug_config file: {e}")

    # Build dynamic model folder name
    model_folder_name = f"model_{MODEL_TYPE}_env-{aug_params['environment']}_color-{aug_params['color_mode']}_noise-{aug_params['noise']}_aug-{aug_params['aug_per_img']}_size-{aug_params['img_size']}"
    model_dir = os.path.join(base_dir, model_folder_name)
    model_path = os.path.join(model_dir, "isonet_final.pth")
    eval_output_dir = os.path.join(model_dir, "evaluation")
    split_path = os.path.join(model_dir, "split_indices.json")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_output_dir, exist_ok=True)
    eval_output_dir = os.path.join(model_dir, "evaluation")
    split_path = os.path.join(model_dir, "split_indices.json")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_output_dir, exist_ok=True)

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

    # ==== Dataset Split ====
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

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    # ==== Model, Loss, Optimizer ====
    model = get_model(MODEL_TYPE, num_classes)
    if torch.cuda.device_count() > 1:
        print(f"🧠 Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    force_loss_fn = nn.MSELoss()
    class_loss_fn = nn.CrossEntropyLoss()
    point_loss_fn = nn.MSELoss()

    # ==== Resume from latest checkpoint ====
    checkpoint_pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth")
    checkpoint_files = []
    for f in os.listdir(model_dir):
        match = checkpoint_pattern.match(f)
        if match:
            epoch_num = int(match.group(1))
            checkpoint_files.append((epoch_num, os.path.join(model_dir, f)))

    if checkpoint_files:
        checkpoint_files.sort()
        for ep, path in checkpoint_files:
            print(f"📂️ Found checkpoint: {path} (epoch {ep})")

        start_epoch, latest_checkpoint = checkpoint_files[-1]
        print(f"🔁 Loading checkpoint from epoch {start_epoch}: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_losses = checkpoint.get("train_loss", [])
        val_losses = checkpoint.get("val_loss", [])
        print(f"🔁 Resuming training at epoch {start_epoch+1}/{NUM_EPOCHS}")

        # ✅ Skip training if already at final epoch
        if start_epoch == NUM_EPOCHS:
            print(f"⏩ Training already completed up to epoch {NUM_EPOCHS}. Skipping training.")
            run_final_evaluation = True
        else:
            run_final_evaluation = False
    else:
        start_epoch = 0
        train_losses, val_losses = [], []
        run_final_evaluation = False
        print("📅 No checkpoint found — starting training from scratch")

    # ==== Training Loop ====
    if not run_final_evaluation:
        for epoch in range(start_epoch, NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            for images, forces, shape_class, contact_point in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                images = images.to(DEVICE)
                forces = forces.to(DEVICE)
                shape_class = shape_class.to(DEVICE)
                contact_point = contact_point.to(DEVICE)

                # Round Fz and contact points to 1 decimals
                forces[:, 2] = (forces[:, 2] * 10).round() / 10
                contact_point[:, 0] = (contact_point[:, 0] * 10).round() / 10
                contact_point[:, 1] = (contact_point[:, 1] * 10).round() / 10

                optimizer.zero_grad()
                pred_forces, pred_class, pred_point = model(images)

                loss = force_loss_fn(pred_forces[:, 2], forces[:, 2]) + class_loss_fn(pred_class, shape_class)

                valid_mask = ~torch.isnan(contact_point).any(dim=1)
                if valid_mask.any():
                    loss += point_loss_fn(pred_point[valid_mask], contact_point[valid_mask])

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train = train_loss / len(train_loader)
            train_losses.append(avg_train)

            # ==== Validation ====
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, forces, shape_class, contact_point in val_loader:
                    images = images.to(DEVICE)
                    forces = forces.to(DEVICE)
                    shape_class = shape_class.to(DEVICE)
                    contact_point = contact_point.to(DEVICE)

                    forces[:, 2] = (forces[:, 2] * 10).round() / 10
                    contact_point[:, 0] = (contact_point[:, 0] * 10).round() / 10
                    contact_point[:, 1] = (contact_point[:, 1] * 10).round() / 10

                    pred_forces, pred_class, pred_point = model(images)
                    loss = force_loss_fn(pred_forces[:, 2], forces[:, 2]) + class_loss_fn(pred_class, shape_class)

                    valid_mask = ~torch.isnan(contact_point).any(dim=1)
                    if valid_mask.any():
                        loss += point_loss_fn(pred_point[valid_mask], contact_point[valid_mask])

                    val_loss += loss.item()

            avg_val = val_loss / len(val_loader)
            val_losses.append(avg_val)
            print(f"📉 Epoch {epoch+1} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

            # ==== Save Checkpoint every 2 epochs ====
            if (epoch + 1) % 2 == 0 or (epoch + 1 == NUM_EPOCHS):
                checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_losses,
                    'val_loss': val_losses
                }, checkpoint_path)
                print(f"📀 Checkpoint saved: {checkpoint_path}")


            # ==== Save Final Model (only if full training completed) ====
            if (epoch + 1) == NUM_EPOCHS:
                torch.save(model.state_dict(), model_path)
                print(f"✅ Final model saved to: {model_path}")

    # ==== Plot Training ====
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(model_dir, "training_loss_plot.png"))
    plt.close()
    print(f"📈 Loss plot saved to: {os.path.join(model_dir, 'training_loss_plot.png')}")

    # ==== Final Evaluation ====
    print("\n🧪 Final evaluation on test set...")
    model.eval()

    test_force_loss, test_class_loss, test_point_loss = 0.0, 0.0, 0.0
    force_true_all, force_pred_all = [], []
    point_true_all, point_pred_all = [], []
    y_true_cls, y_pred_cls = [], []

    with torch.no_grad():
        for images, forces, shape_class, contact_point in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            forces = forces.to(DEVICE)
            shape_class = shape_class.to(DEVICE)
            contact_point = contact_point.to(DEVICE)

            # Round to sensor resolution
            forces[:, 2] = forces[:, 2].round(decimals=2)  # Fz
            contact_point[:, 0] = contact_point[:, 0].round(decimals=3)
            contact_point[:, 1] = contact_point[:, 1].round(decimals=3)

            pred_forces, pred_class, pred_point = model(images)

            # Losses
            test_force_loss += force_loss_fn(pred_forces[:, 2], forces[:, 2]).item() * images.size(0)
            test_class_loss += class_loss_fn(pred_class, shape_class).item() * images.size(0)

            valid_mask = ~torch.isnan(contact_point).any(dim=1)
            if valid_mask.any():
                test_point_loss += point_loss_fn(pred_point[valid_mask], contact_point[valid_mask]).item() * valid_mask.sum().item()

            # Store predictions
            force_true_all.append(forces[:, 2].cpu())
            force_pred_all.append(pred_forces[:, 2].cpu())
            point_true_all.append(contact_point.cpu())
            point_pred_all.append(pred_point.cpu())

            pred_class_argmax = pred_class.argmax(dim=1)
            y_true_cls += shape_class.cpu().tolist()
            y_pred_cls += pred_class_argmax.cpu().tolist()

    # === Compute final MSE for valid contact points ===
    point_true_all = torch.cat(point_true_all, dim=0)
    point_pred_all = torch.cat(point_pred_all, dim=0)
    valid_mask = ~torch.isnan(point_true_all).any(dim=1)
    if valid_mask.any():
        final_point_mse = point_loss_fn(point_pred_all[valid_mask], point_true_all[valid_mask]).item()
    else:
        final_point_mse = float("nan")

    n = len(test_set)
    print(f"\n📊 Final TEST metrics over {n} samples:")
    print(f"  🔧 Force MSE (Fz only): {test_force_loss / n:.4f}")
    print(f"  🎯 Class CE Loss       : {test_class_loss / n:.4f}")
    print(f"  📍 Point MSE (x,y)     : {final_point_mse:.4f}")

    # === Classification Metrics ===
    target_names = [str(label) for label in dataset.get_shape_labels()]
    print("\n🎯 Classification Report:")
    print(classification_report(y_true_cls, y_pred_cls, target_names=target_names))
    print(f"  ✅ Accuracy : {accuracy_score(y_true_cls, y_pred_cls):.4f}")
    print(f"  ✅ Precision: {precision_score(y_true_cls, y_pred_cls, average='macro'):.4f}")
    print(f"  ✅ Recall   : {recall_score(y_true_cls, y_pred_cls, average='macro'):.4f}")
    print(f"  ✅ F1 Score : {f1_score(y_true_cls, y_pred_cls, average='macro'):.4f}")

    # === Save metrics summary to JSON ===
    metrics = {
        "force_mse_fz": round(test_force_loss / n, 4),
        "class_ce_loss": round(test_class_loss / n, 4),
        "point_mse_xy": round(final_point_mse, 4) if not np.isnan(final_point_mse) else None,
        "accuracy": round(accuracy_score(y_true_cls, y_pred_cls), 4),
        "precision_macro": round(precision_score(y_true_cls, y_pred_cls, average='macro'), 4),
        "recall_macro": round(recall_score(y_true_cls, y_pred_cls, average='macro'), 4),
        "f1_macro": round(f1_score(y_true_cls, y_pred_cls, average='macro'), 4)
    }

    with open(os.path.join(eval_output_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"📊 Saved metrics summary to: {os.path.join(eval_output_dir, 'metrics_summary.json')}")


    # === Confusion Matrix ===
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_output_dir, "confusion_matrix.png"))
    plt.close()

    print(f"\n✅ All evaluation metrics and confusion matrix saved to: {eval_output_dir}")

if __name__ == "__main__":
    main()
    print("🚀 Training and evaluation completed successfully!")