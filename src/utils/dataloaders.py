"""
This module handles data loading and preprocessing for CNN models.

Key features:
- PlantDiseaseDataset: A custom PyTorch Dataset that reads from M1 mapped CSVs.
- DataLoader factory functions for training, validation, and testing sets.
"""

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple


class PlantDiseaseDataset(Dataset):
    """
    Dataset for Plant Disease Classification.
    Reads from a mapped CSV split file (generated in M1).
    Expected columns: 'filepath_rel', 'canonical_id'
    """

    def __init__(self, csv_filepath: str, root_dir: str, transform=None):
        self.data = pd.read_csv(csv_filepath)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Path is relative to project root
        img_path = self.root_dir / row["filepath_rel"]
        label = int(row["canonical_id"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise e

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_dataloader(
    train_csv: str,
    root_dir: str = ".",
    batch_size: int = 32,
    num_workers: int = 4,
    transforms: Optional[transforms.Compose] = None,
    collate_fn=None
) -> DataLoader:
    """
    Creates DataLoaders for training and validation with appropriate data augmentation.

    This function sets up the data pipeline for model training:

    Args:
        train_csv: Path to training split CSV file
        val_csv: Path to validation split CSV file
        root_dir: Root directory for image paths (default: current directory)
        batch_size: Number of samples per batch (default: 32)
        img_size: Target image size for resizing (default: 224)
        num_workers: Number of worker processes for data loading (default: 4)
        transforms: Optional custom transforms to apply.

    Returns:
        train_loader: PyTorch DataLoader for training data
    """

    train_dataset = PlantDiseaseDataset(train_csv, root_dir, transform=transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader

def get_val_dataloader(
    val_csv: str,
    root_dir: str = ".",
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    transforms: Optional[transforms.Compose] = None,
    collate_fn=None
    )-> DataLoader:
    """
    Creates DataLoader for validation.

    Args:
        val_csv: Path to validation split CSV file
        root_dir: Root directory for image paths (default: current directory)
        batch_size: Number of samples per batch (default: 32)
        img_size: Target image size for resizing (default: 224)
        num_workers: Number of worker processes for data loading (default: 4)
        transforms: Optional custom transforms to apply.

    Returns:
        val_loader: PyTorch DataLoader for validation data
    """

    val_dataset = PlantDiseaseDataset(val_csv, root_dir, transform=transforms)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return val_loader


def get_test_dataloader(
    test_csv: str,
    root_dir: str = ".",
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    transforms: Optional[transforms.Compose] = None,
    collate_fn=None
) -> DataLoader:
    """
    Creates DataLoader for testing.

    Args:
        test_csv: Path to test split CSV file
        root_dir: Root directory for image paths (default: current directory)
        batch_size: Number of samples per batch (default: 32)
        img_size: Target image size for resizing (default: 224)
        num_workers: Number of worker processes for data loading (default: 4)
        transforms: Optional custom transforms to apply.

    Returns:
        test_loader: PyTorch DataLoader for test data
    """

    test_dataset = PlantDiseaseDataset(test_csv, root_dir, transform=transforms)

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
