import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from pypylon import pylon
from PIL import Image, ImageDraw

# ==== USER CONFIGURATION SECTION ====
MODEL_TYPE = "resnet18"  # "resnet18" or "owncnn"
COLOR_MODE = "grayscale"  # "grayscale" or "rgb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\code\code full pipeline\All code\Code from laptop\Testing_data_del\Data\full_dataset"
MODEL_PATH = os.path.join(BASE_DIR, "model_checkpoint.pth")
LABELS_CSV = os.path.join(BASE_DIR, "labels.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MAX_POINTS = 100

# ==== Dynamic Imports ====
def get_model(model_type, num_shape_classes):
    if model_type == "resnet18":
        from model_ResNet18 import IsoNet as SelectedNet
    elif model_type == "owncnn":
        from model_own_CNN import IsoNet as SelectedNet
    else:
        raise ValueError("Unknown MODEL_TYPE: choose 'resnet18' or 'owncnn'")
    return SelectedNet(num_shape_classes=num_shape_classes)

def get_dataset():
    from inference_dataset import IsochromaticDataset
    return IsochromaticDataset(images_folder=IMAGE_DIR, labels_csv=LABELS_CSV)

# ==== Preprocessing ====
def get_preprocessor(color_mode):
    if color_mode == "grayscale":
        def preprocess(img_bgr):
            img = Image.fromarray(img_bgr).convert("L")
            img = img.crop((30, 0, img.width, img.height))
            left = (img.width - 980) // 2
            top = (img.height - 980) // 2
            img = img.crop((left, top, left + 980, top + 980))
            mask = Image.new("L", (980, 980), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, 980, 980), fill=255)
            img_aug = Image.composite(img, Image.new("L", img.size), mask)
            img_aug = img_aug.resize((224, 224))
            img_tensor = transforms.ToTensor()(img_aug)
            img_tensor = transforms.Normalize(mean=[0.5], std=[0.5])(img_tensor)
            img_tensor = img_tensor.expand(3, -1, -1)
            return img_tensor
        return preprocess
    elif color_mode == "rgb":
        def preprocess(img_bgr):
            img_rgb = Image.fromarray(img_bgr[..., ::-1])
            img_rgb = img_rgb.crop((30, 0, img_rgb.width, img_rgb.height))
            left = (img_rgb.width - 980) // 2
            top = (img_rgb.height - 980) // 2
            img_rgb = img_rgb.crop((left, top, left + 980, top + 980))
            mask = Image.new("L", (980, 980), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, 980, 980), fill=255)
            img_aug = Image.composite(img_rgb, Image.new("RGB", img_rgb.size), mask)
            img_aug = img_aug.resize((224, 224))
            img_tensor = transforms.ToTensor()(img_aug)
            img_tensor = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(img_tensor)
            return img_tensor
        return preprocess
    else:
        raise ValueError("Unknown COLOR_MODE: choose 'grayscale' or 'rgb'")

# ==== Main Inference Logic ====
def main():
    print(f"🖥️ Using device: {DEVICE}")
    print(f"Model: {MODEL_TYPE}, Color mode: {COLOR_MODE}")
    dataset = get_dataset()
    shape_labels = dataset.get_shape_labels()
    model = get_model(MODEL_TYPE, num_shape_classes=len(shape_labels)).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    preprocess = get_preprocessor(COLOR_MODE)

    # Camera setup
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.AcquisitionFrameRateEnable.SetValue(True)
    print("Camera initialized.")

    # Inference loop (single frame for demo)
    grab_result = camera.GrabOne(1000)
    if grab_result.GrabSucceeded():
        img_bgr = grab_result.Array
        img_tensor = preprocess(img_bgr).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_forces, pred_class, pred_point = model(img_tensor)
        print(f"Predicted forces: {pred_forces}")
        print(f"Predicted class: {pred_class.argmax(dim=1)}")
        print(f"Predicted contact point: {pred_point}")
    else:
        print("Failed to grab image from camera.")
    camera.Close()

if __name__ == "__main__":
    main()
