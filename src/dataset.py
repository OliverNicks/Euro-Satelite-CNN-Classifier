"""Dataset loaders, preprocessing, and image augmentation transforms for EuroSAT.
"""

from typing import Callable, Dict, List, Optional, Tuple
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet normalization standard (used in trained model)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (64, 64)


def get_train_transforms() -> transforms.Compose:
    """Data augmentation pipeline for training."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.15, contrast=0.15)],
            p=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """Standard evaluation and inference transform (deterministic)."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


class SateliteDataset(Dataset):
    """PyTorch Dataset for labeled satellite imagery."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = row["file_path"]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label"])
        return image, label


class TestSatelliteDataset(Dataset):
    """PyTorch Dataset for unlabeled test images (e.g. public_test_data)."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        row = self.df.iloc[idx]
        img_path = os.path.join(row["folder"], row["file_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, row["file_name"]


def preprocess_dataset(data_folder: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Recursively scan data folder for class subdirectories and build DataFrame."""
    image_data: List[Dict[str, object]] = []
    labels: Dict[str, int] = {}
    label_counter = 0

    # Ensure deterministic alphabetical sorting of classes
    sorted_dirs = sorted([
        d for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d))
    ])

    for class_name in sorted_dirs:
        if class_name not in labels:
            labels[class_name] = label_counter
            label_counter += 1

        class_dir = os.path.join(data_folder, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                image_data.append({
                    "file": fname,
                    "file_path": os.path.join(class_dir, fname),
                    "label": labels[class_name],
                    "class_name": class_name,
                })

    df = pd.DataFrame(image_data)
    return df, labels


def preprocess_test_folder(folder_path: str) -> pd.DataFrame:
    """Build DataFrame for unlabeled test images in a flat folder."""
    files = [
        f for f in sorted(os.listdir(folder_path))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return pd.DataFrame({"file_name": files, "folder": folder_path})
