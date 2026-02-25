"""
Model Training Script

This module trains models for plant disease classification.
It handles the complete training pipeline including:
- Data loading from M1 split CSVs
- Model initialization with pre-trained weights
- Training loop with validation
- Checkpoint saving (best model based on validation accuracy)
- Training metrics logging to CSV

Usage:
    python src/train/train.py --config configs/baseline_mobilenet_v3_small.json
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import sys
import ssl
from pathlib import Path
import platform
from tqdm import tqdm
import json
import random
import numpy as np

# Add project root to path BEFORE importing local modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import helpers for data and model loading
from src.utils.dataloaders import get_train_dataloader, get_val_dataloader
from src.utils.baseline_models import get_model
from src.utils.transformations import get_transforms, get_collate_fn

# Only runs if on MacOS (Darwin is the OS kernel name for MacOS)
# Disable SSL verification to fix for MacOS SSL error when downloading models
if platform.system() == "Darwin":
    ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Crucial for deterministic behavior on some hardware
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For MacOS (MPS)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one complete training epoch over the entire training dataset.

    This function:
    - Sets the model to training mode (enables dropout, batch norm updates)
    - Iterates through all batches in the training loader
    - Performs forward pass, loss calculation, backpropagation, and parameter updates
    - Displays real-time progress with tqdm showing loss and accuracy

    Args:
        model: PyTorch model to train
        loader: DataLoader containing training data
        criterion: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer for parameter updates (e.g., Adam)
        device: Device to run computations on (cpu/cuda/mps)

    Returns:
        Tuple of (epoch_loss, epoch_acc) - average loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass (predictions)
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass (calculate gradients) and optimize
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item() * images.size(0)

        # Get predictions and update total
        _, predicted = outputs.max(1)
        
        if labels.ndim == 2:  # mixup/cutmix case
            hard_labels = labels.argmax(dim=1)
        else:
            hard_labels = labels

        total += hard_labels.size(0)

        # Count how many predictions are correct
        correct += predicted.eq(hard_labels).sum().item()

        # Update progress bar
        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation dataset.

    This function:
    - Sets the model to evaluation mode (disables dropout, freezes batch norm)
    - Disables gradient computation for efficiency (torch.no_grad)
    - Computes predictions and metrics without updating model parameters
    - Displays progress with tqdm

    Args:
        model: PyTorch model to evaluate
        loader: DataLoader containing validation data
        criterion: Loss function for computing validation loss
        device: Device to run computations on (cpu/cuda/mps)

    Returns:
        Tuple of (epoch_loss, epoch_acc) - average loss and accuracy on validation set
    """
    # Set model to evaluation mode, freeze gradients and disable dropout
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculation
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass (predictions)
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Update running loss
            running_loss += loss.item() * images.size(0)

            # Get predictions and update total
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    # Set seed for reproducibility
    set_seed()

    parser = argparse.ArgumentParser(
        description="Train Models for Plant Disease Classification"
    )
    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     default="mobilenet_v3_small",
    #     choices=["mobilenet_v3_small", "efficientnet_b0", "vit_base_patch16_224"],
    # )
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--data-dir", type=str, default=".")
    # parser.add_argument("--splits-dir", type=str, default="data/splits")
    # parser.add_argument("--output-dir", type=str, default="outputs")
    # parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--debug", action="store_true", help="Run with small subset")

    args = parser.parse_args()

    # Load config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # Deserialize paths and settings
    output_dir = config.get("output_dir", "outputs")
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    data_dir = config.get("data_dir", ".")
    splits_dir = config.get("splits_dir", "data/splits")
    model_name = config["model_name"]  # should throw an error if missing

    # Deserialize hyperparameters
    epochs = config["hyperparameters"].get(
        "num_epochs", config["hyperparameters"].get("epochs", 10)
    )
    batch_size = config["hyperparameters"].get("batch_size", 32)
    lr = config["hyperparameters"].get("learning_rate", 0.001)
    weight_decay = config["hyperparameters"].get("weight_decay", 0.0)

    # Deserialize transformations config
    transform_config = config.get("transformations", None)

    # Setup directories 
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Device switching to utilize mps/gpu if available, otherwise use CPU
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Data Paths
    train_csv = Path(splits_dir) / "pv_train.csv"
    val_csv = Path(splits_dir) / "pv_val.csv"

    # Check if split partitions exist
    if not train_csv.exists() or not val_csv.exists():
        print(f"Error: Split partitions not found in {splits_dir}")
        print("Please run M1 pipeline first.")
        return

    # Model
    print(f"Initializing {model_name}...")
    model = get_model(model_name, num_classes=26)
    model.to(device)

    # Get data transforms based on model
    train_transform, val_transform, test_transform = get_transforms(
        model=model,
        model_name=model_name,
        image_size=224,
        transforms_config=transform_config
    )

    # Get custom collate function if CutMix or MixUp is specified
    train_collate_fn = get_collate_fn(
        transforms_config=transform_config
    )

    # Load Data
    print(f"Loading data from {splits_dir}...")
    train_loader = get_train_dataloader(
        train_csv, root_dir=data_dir, batch_size=batch_size, transforms=train_transform, collate_fn=train_collate_fn,
        num_workers=args.num_workers
    )
    val_loader = get_val_dataloader(
        val_csv, root_dir=data_dir, batch_size=batch_size, transforms=val_transform,
        num_workers=args.num_workers
    )

    # Debug mode: truncate datasets
    if args.debug:
        print("DEBUG MODE: Truncating datasets")
        train_loader.dataset.data = train_loader.dataset.data.head(100)
        val_loader.dataset.data = val_loader.dataset.data.head(20)

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training Loop
    best_val_acc = 0.0
    history = []

    # Start timer
    start_time = time.time()

    # Training Loop
    for epoch in range(epochs):
        ep_start = time.time()

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate for one epoch
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print epoch results
        dt = time.time() - ep_start
        print(
            f"Epoch {epoch+1}/{epochs} [{dt:.1f}s] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Append to history
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time": dt,
            }
        )

        # Save best model to checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = Path(checkpoint_dir) / f"{Path(args.config).stem}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  --> Saved new best model (Acc: {best_val_acc:.4f})")

    # Print total training time
    total_time = time.time() - start_time
    print(
        f"\nTraining complete in {total_time/60:.2f}m. Best Val Acc: {best_val_acc:.4f}"
    )

    # Save logs
    log_path = Path(output_dir) / f"training_log_{Path(args.config).stem}.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"Logs saved to {log_path}")


if __name__ == "__main__":
    main()
