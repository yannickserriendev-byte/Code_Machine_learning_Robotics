import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw
from pypylon import pylon

from Code_Machine_learning_Robotics.inference.inference_dataset import IsochromaticDataset
from Code_Machine_learning_Robotics.inference.model_ResNet18 import IsoNet

MODEL_BASE_DIR = r"Y:\Master Thesis\Sensor_1\Crisp data\Data\full_dataset"
MODEL_PATH = os.path.join(MODEL_BASE_DIR, "model_ResNet18_Sensor_1_Crisp_no_zip_3aug_per_img_super_computer_corrected_rotation", "checkpoint_epoch_40.pth")
BASE_DIR = r"Y:\Master Thesis\Sensor_1\Crisp data\Data\full_dataset"
LABELS_CSV = os.path.join(BASE_DIR, "3.augmented_labels_full_pipeline_rotation_fix_ShapeWithNone_XYWithNan_based_on_Fz_indentorclassfix.csv")
DEVICE = torch.device("cpu")
MAX_POINTS = 100

dataset = IsochromaticDataset(
    images_folder=os.path.join(BASE_DIR, "1.augmented_images_full_pipeline_3_aug_per_img"),
    labels_csv=LABELS_CSV,
)
dataset.data = pd.read_csv(LABELS_CSV, keep_default_na=False, na_values=[])
shape_index_to_label = {
    int(cls): str(name)
    for name, cls in sorted(
        dataset.data[['Indentor_Shape', 'shape_class']].drop_duplicates().values.tolist(),
        key=lambda x: x[1]
    )
}
shape_labels = [shape_index_to_label[i] for i in sorted(shape_index_to_label)]
print(f"📦 Loaded shape label mapping: {shape_labels}")

model = IsoNet(num_shape_classes=len(shape_labels)).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
raw_state_dict = checkpoint["model_state_dict"]
new_state_dict = {k.replace("module.", ""): v for k, v in raw_state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

class CustomPreprocessor:
    def __init__(self, shift_px=30, crop_size=980, final_size=224):
        self.shift_px = shift_px
        self.crop_size = crop_size
        self.final_size = final_size

    def __call__(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # ✔ Convert to RGB
        img = Image.fromarray(img_rgb)

        img = img.crop((self.shift_px, 0, img.width, img.height))

        # Center crop
        left = (img.width - self.crop_size) // 2
        top = (img.height - self.crop_size) // 2
        img = img.crop((left, top, left + self.crop_size, top + self.crop_size))

        # Circular mask
        mask = Image.new("L", (self.crop_size, self.crop_size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, self.crop_size, self.crop_size), fill=255)
        img_aug = Image.composite(img, Image.new("RGB", img.size), mask)

        # Resize and convert to tensor
        img_resized = img_aug.resize((self.final_size, self.final_size))
        tensor = TF.to_tensor(img_resized)
        tensor = TF.normalize(tensor, mean=[0.5]*3, std=[0.5]*3)

        return tensor, img_aug  # img_aug remains unnormalized for display



transform = CustomPreprocessor(shift_px=30, crop_size=980, final_size=224)

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.AcquisitionFrameRateEnable.SetValue(True)
camera.AcquisitionFrameRate.SetValue(50.0)
camera.ExposureAuto.SetValue("Off")
camera.ExposureTime.SetValue(2000.0)
camera.GainAuto.SetValue("Off")
camera.Gain.SetValue(0)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

plt.ion()
fig = plt.figure(figsize=(14, 6))
gs = GridSpec(2, 3, width_ratios=[1.5, 1.5, 1], height_ratios=[1, 2])

ax_img = fig.add_subplot(gs[:, 0])
im_display = ax_img.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
ax_img.set_title("Camera Feed")
ax_img.axis("off")

ax_img_aug = fig.add_subplot(gs[:, 1])
im_pre_display = ax_img_aug.imshow(np.zeros((224, 224)), cmap='gray', vmin=0, vmax=255)
ax_img_aug.set_title("Masked Input (RGB)")
ax_img_aug.axis("off")

ax_text = fig.add_subplot(gs[0, 2])
ax_text.axis("off")
text_shape = ax_text.text(0.0, 0.8, "", fontsize=12)
text_contact = ax_text.text(0.0, 0.5, "", fontsize=12)
text_fz = ax_text.text(0.0, 0.2, "", fontsize=12)

ax_force = fig.add_subplot(gs[1, 2])
fz_line, = ax_force.plot([], [], label="Fz (Normal)")
ax_force.set_ylim(0, 20)
ax_force.set_xlim(0, MAX_POINTS)
ax_force.set_xlabel("Time step")
ax_force.set_ylabel("Force (N)")
ax_force.set_title("Fz vs Time")
ax_force.legend()
ax_force.grid(True)

force_fz_vals = []
fps_window = []

stop_flag = {"value": False}

def on_key(event):
    if event.key == 'q':
        print("\n⛔ Quit key pressed.")
        stop_flag["value"] = True

fig.canvas.mpl_connect("key_press_event", on_key)

try:
    while camera.IsGrabbing() and not stop_flag["value"]:
        tic = time.time()
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            grab.Release()
            continue
        frame_bgr = converter.Convert(grab).GetArray()
        grab.Release()
        input_tensor, img_aug = transform(frame_bgr)
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_forces, pred_class_logits, pred_contact = model(input_tensor)

        fz_val = float(pred_forces.squeeze()[2].cpu().item())
        shape_idx = int(pred_class_logits.argmax().item())
        shape_lbl = shape_labels[shape_idx] if shape_idx < len(shape_labels) else f"Class {shape_idx}"
        contact_xy = pred_contact.squeeze().cpu().numpy()

        im_display.set_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        im_pre_display.set_data(np.array(img_aug))
        text_shape.set_text(f"Shape: {shape_lbl}")
        if shape_lbl.lower() == 'none':
            text_contact.set_text("Contact: -")
        else:
            text_contact.set_text(f"Contact: (x={contact_xy[0]:.1f}, y={contact_xy[1]:.1f})")
        text_fz.set_text(f"Fz: {fz_val:.2f} N")

        force_fz_vals.append(fz_val)
        if len(force_fz_vals) > MAX_POINTS:
            force_fz_vals.pop(0)
        fz_line.set_data(range(len(force_fz_vals)), force_fz_vals)
        ax_force.set_xlim(0, max(MAX_POINTS, len(force_fz_vals)))

        plt.pause(0.001)
        fps_window.append(1.0 / (time.time() - tic))
        if len(fps_window) > 100:
            fps_window.pop(0)

finally:
    camera.StopGrabbing()
    camera.Close()
    plt.ioff()
    plt.close()
    if fps_window:
        avg_fps = sum(fps_window) / len(fps_window)
        print(f"\n📉 Average model labelling time: {1.0 / avg_fps:.3f} sec (\u2248 {avg_fps:.2f} FPS)")
