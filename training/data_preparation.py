import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import matplotlib.pyplot as plt

class IsochromaticDataset(Dataset):
    """
    PyTorch Dataset for loading photoelastic fringe pattern images and associated labels.
    Used for training multi-task tactile sensing models.
    """
    def __init__(self, images_folder, labels_csv, transform=None):
        """
        Initialize IsochromaticDataset for training.
        Args:
            images_folder (str): Path to folder containing images.
            labels_csv (str): Path to CSV file with labels and metadata.
            transform (callable, optional): Image transform pipeline.
        """
        self.images_folder = images_folder
        self.data = pd.read_csv(labels_csv)
        # Clean string columns
        # self.data = self.data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        self.data = pd.read_csv(labels_csv)
        # Ensure numeric types where needed
        numeric_cols = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Ft', 'X_Position_mm_after_rotation', 'Y_Position_mm_after_rotation']
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset by index.
        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: (image tensor, forces tensor, shape class tensor, contact point tensor)
        """
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_folder, row['New_Image_Name'])
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"\u274c Missing image: {img_path}")
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        forces = torch.tensor(row[['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Ft']].astype(np.float32).values)
        shape_class = torch.tensor(row['shape_class'], dtype=torch.long)
        contact_point = torch.tensor(row[['X_Position_mm_after_rotation', 'Y_Position_mm_after_rotation']].astype(np.float32).values)
        return image, forces, shape_class, contact_point

    def get_shape_labels(self):
        """
        Return sorted list of unique shape class labels.
        """
        return sorted(self.data['shape_class'].dropna().unique().astype(int))
