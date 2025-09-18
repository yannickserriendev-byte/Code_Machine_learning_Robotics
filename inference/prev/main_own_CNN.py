# Live acquisition + inference GUI (Fz + shape + contact)
# -------------------------------------------------------------
# This version assumes the neural network returns **three tensors** in the
# following order:
#   1. `pred_forces`           – shape (1,7)          → force vector, index 2 is Fz (N)
#   2. `pred_class_logits`     – shape (1, N_shapes)   → soft‑max for object shape
#   3. `pred_contact`          – shape (1, 2)          → x, y contact point (pixels or mm)
# -------------------------------------------------------------

import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision import transforms
from pypylon import pylon

# ── Project‑specific imports ────────────────────────────────────────────
from Code_Machine_learning_Robotics.inference.inference_dataset import IsochromaticDataset
from model_ResNet18 import IsoNet

# ==== Configuration =====================================================
BASE_DIR = r"C:\aa TU Delft\2. Master BME TU Delft + Rheinmetall Internship + Harvard Thesis\2. Year 2\2. Master Thesis at TU Delft\3. Master Thesis\2. Data creation\data aqcuisition\1\full_dataset"
MODEL_PATH = os.path.join(BASE_DIR, "..", "model_4_ResNet18_2shapes_less_rainbows_sensor", "isonet_final.pth")
LABELS_CSV = os.path.join(BASE_DIR, "3.augmented_labels_scaled_filtered_indentor_and_xy_values_to_None_and_Nan_y_value_corrected.csv")
DEVICE = torch.device("cpu")        # Change to "cuda" if a GPU is available
MAX_POINTS = 100                     # Points visible in the Fz plot

# ==== Load shape‑label mapping ==========================================
dataset = IsochromaticDataset(
    images_folder=os.path.join(BASE_DIR, "3.augmented_images"),
    labels_csv=LABELS_CSV,
)
shape_labels = dataset.get_shape_labels()  # list[str]

# ==== Load trained model ===============================================
# Force shape class count to 2 to match the trained model
model = IsoNet(num_shape_classes=2).to(DEVICE)

# Handle DataParallel checkpoints by stripping "module." prefix
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint if "state_dict" not in checkpoint else checkpoint["state_dict"]
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# ==== Image transform ===================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

# ==== Basler camera initialisation =====================================
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.AcquisitionFrameRateEnable.SetValue(True)
camera.AcquisitionFrameRate.SetValue(25.0)
camera.ExposureAuto.SetValue("Off")
camera.ExposureTime.SetValue(2000.0)  # µs
camera.GainAuto.SetValue("Off")
camera.Gain.SetValue(0)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# ==== Matplotlib GUI setup =============================================
plt.ion()
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 2])

# Camera image -----------------------------------------------------------
ax_img = fig.add_subplot(gs[:, 0])
im_display = ax_img.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
ax_img.set_title("Camera Feed")
ax_img.axis("off")

# Text read‑out ----------------------------------------------------------
ax_text = fig.add_subplot(gs[0, 1])
ax_text.axis("off")
text_shape   = ax_text.text(0.0, 0.8, "", fontsize=12)
text_contact = ax_text.text(0.0, 0.5, "", fontsize=12)
text_fz      = ax_text.text(0.0, 0.2, "", fontsize=12)

# Fz plot ----------------------------------------------------------------
ax_force = fig.add_subplot(gs[1, 1])
fz_line, = ax_force.plot([], [], label="Fz (Normal)")
ax_force.set_ylim(0, 20)           # Adjust as needed to your sensor range
ax_force.set_xlim(0, MAX_POINTS)
ax_force.set_xlabel("Time step")
ax_force.set_ylabel("Force (N)")
ax_force.set_title("Fz vs Time")
ax_force.legend()
ax_force.grid(True)

force_fz_vals = []  # rolling window of Fz predictions

# ==== Main acquisition & inference loop ================================
fps_window = []
while camera.IsGrabbing():
    tic = time.time()

    grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if not grab.GrabSucceeded():
        grab.Release()
        continue  # skip faulty frame

    # 1. Convert to numpy BGR image
    frame_bgr = converter.Convert(grab).GetArray()
    grab.Release()

    # 2. Pre‑process & inference
    input_tensor = transform(frame_bgr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_forces, pred_class_logits, pred_contact = model(input_tensor)

    # --- Post‑process outputs -----------------------------------------
    fz_val    = float(pred_forces.squeeze()[2].cpu().item())
    shape_idx = int(pred_class_logits.argmax().item())
    shape_lbl = shape_labels[shape_idx] if shape_idx < len(shape_labels) else f"Class {shape_idx}"
    contact_xy = pred_contact.squeeze().cpu().numpy()  # (2,)

    # --- Update GUI ---------------------------------------------------
    # Image
    im_display.set_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    # Text
    text_shape.set_text(f"Shape: {shape_lbl}")
    text_contact.set_text(f"Contact: (x={contact_xy[0]:.1f}, y={contact_xy[1]:.1f})")
    text_fz.set_text(f"Fz: {fz_val:.2f} N")

    # Fz time‑series plot
    force_fz_vals.append(fz_val)
    if len(force_fz_vals) > MAX_POINTS:
        force_fz_vals.pop(0)
    fz_line.set_data(range(len(force_fz_vals)), force_fz_vals)
    ax_force.set_xlim(0, max(MAX_POINTS, len(force_fz_vals)))

    plt.pause(0.001)  # allow GUI to refresh

    # --- FPS calculation ---------------------------------------------
    fps_window.append(1.0 / (time.time() - tic))
    if len(fps_window) > 100:
        fps_window.pop(0)

# ==== Clean‑up ==========================================================
camera.StopGrabbing()
camera.Close()
plt.ioff()
plt.close()

# ==== Final statistics ======================