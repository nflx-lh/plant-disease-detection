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

# Add project root to path BEFORE importing local modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import helpers for data and model loading
from src.utils.dataloaders import get_train_dataloader, get_val_dataloader
from src.utils.baseline_models import get_model
from src.utils.transformations import get_transforms

from src.utils.supcon import SupConLoss, SupConViT, get_label_mappings, create_mask, TwoCropTransform

# Only runs if on MacOS (Darwin is the OS kernel name for MacOS)
# Disable SSL verification to fix for MacOS SSL error when downloading models
if platform.system() == "Darwin":
    ssl._create_default_https_context = ssl._create_unverified_context

def unfreeze_backbone(model: SupConViT):
    # Freeze all layers
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False

    if model.backbone_name == "vit_base_patch16_224":
        # Unfreeze last few encoder layers
        for block in model.backbone.blocks[-4:]:
            for param in block.parameters():
                param.requires_grad = True
    elif model.backbone_name == "maxvit_base_tf_224":
        # Unfreeze last stage
        for stage in model.backbone.stages[-1:]:
            for param in stage.parameters():
                param.requires_grad = True

    return model

def unfreeze_classifier(model: SupConViT):
    # Freeze all layers
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False

    if model.backbone_name == "vit_base_patch16_224":
        # Unfreeze classifier
        for param in model.backbone.head.parameters():
            param.requires_grad = True
    elif model.backbone_name == "maxvit_base_tf_224":
        # Unfreeze last stage and the head layers
        for param in model.backbone.head.parameters():
            param.requires_grad = True

    return model


def train_one_epoch_embedding(model: SupConViT, loader, embed_criterion, optimizer, device, canonical_to_crop, canonical_to_disease):
    """
    Executes one complete training epoch over the entire training dataset.

    This function:
    - Sets the model to training mode (enables dropout, batch norm updates)
    - Iterates through all batches in the training loader
    - Performs forward pass, loss calculation, backpropagation, and parameter updates
    - Displays real-time progress with tqdm showing loss

    Args:
        model: PyTorch model to train
        loader: DataLoader containing training data
        criterion: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer for parameter updates (e.g., Adam)
        device: Device to run computations on (cpu/cuda/mps)

    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0

    model = unfreeze_backbone(model)

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = torch.cat([images[0], images[1]], dim=0)
        images, labels = images.to(device), labels.to(device)
        bsz = labels.shape[0]

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass (predictions)
        features, _ = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # Compute loss
        mask = create_mask(labels, canonical_to_crop, canonical_to_disease, device)
        loss = embed_criterion(features, mask=mask)

        # Backward pass (calculate gradients) and optimize
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item() * images.size(0)

        # Update progress bar
        pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def train_one_epoch_classifier(model: SupConViT, loader, classify_criterion, optimizer, device):
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

    model = unfreeze_classifier(model)

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = torch.cat([images[0], images[1]], dim=0)
           
        images, labels = images.to(device), labels.to(device)

        bsz = labels.shape[0]

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass (predictions)
        features, outputs = model(images)
        o1, o2 = torch.split(outputs, [bsz, bsz], dim=0)

        # Compute loss
        loss = classify_criterion(o1, labels) + classify_criterion(o2, labels)

        # Backward pass (calculate gradients) and optimize
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item() * images.size(0)

        # Get predictions and update total
        _, predicted1 = o1.max(1)
        _, predicted2 = o2.max(1)
        total += labels.size(0)

        # Count how many predictions are correct
        correct += predicted1.eq(labels).sum().item()
        correct += predicted2.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model: SupConViT, loader, embed_criterion, classify_criterion, device, canonical_to_crop, canonical_to_disease):
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
            features, outputs = model(images)

            # Compute loss
            loss = classify_criterion(outputs, labels)

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
    parser = argparse.ArgumentParser(description="Train Models for Plant Disease Classification")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
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
    model_name = config["model_name"] # should throw an error if missing
    
    # Deserialize hyperparameters
    epochs = config["hyperparameters"].get("num_epochs", 10)
    batch_size = config["hyperparameters"].get("batch_size", 32)
    lr = config["hyperparameters"].get("learning_rate", 0.001)
    weight_decay = config["hyperparameters"].get("weight_decay", 0.0)

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

    # Get label mappings for hierarchical loss
    crop_to_id, disease_to_id, canonical_to_crop, canonical_to_disease, crop_disease_to_canonical, valid_diseases_per_crop = get_label_mappings(Path(splits_dir) / "label_space.csv")

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
    backbone = get_model(model_name, num_classes=26, pretrained=True)
    backbone.to(device)

    model = SupConViT(backbone, backbone_name=model_name, num_classes=26)
    model.to(device)
    
    # Get data transforms based on model
    train_transform, val_transform, test_transform = get_transforms(
        model=backbone,
        model_name=model_name,
        image_size=224
    )

    train_transform = TwoCropTransform(train_transform)

    # Load Data
    print(f"Loading data from {splits_dir}...")
    train_loader = get_train_dataloader(
        train_csv, root_dir=data_dir, batch_size=batch_size, transforms=train_transform
    )
    val_loader = get_val_dataloader(
        val_csv, root_dir=data_dir, batch_size=batch_size, transforms=val_transform
    )

    # Debug mode: truncate datasets
    if args.debug:
        print("DEBUG MODE: Truncating datasets")
        train_loader.dataset.data = train_loader.dataset.data.head(100)
        val_loader.dataset.data = val_loader.dataset.data.head(20)

    # Optimization
    classify_criterion = nn.CrossEntropyLoss()
    embed_criterion = SupConLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training Loop
    best_val_acc = 0.0
    best_embed_train_loss = float("inf")
    best_embed_model_state_dict = None
    history = []

    # Start timer
    start_time = time.time()

    # Training Loop
    if not (Path(checkpoint_dir) / f"{Path(args.config).stem}_embed.pt").exists():
        print("Train embedding...")
        for epoch in range(epochs):
            ep_start = time.time()

            # Train for one epoch
            train_loss = train_one_epoch_embedding(model, train_loader, embed_criterion, optimizer, device, canonical_to_crop, canonical_to_disease)

            # Print epoch results
            dt = time.time() - ep_start
            print(
                f"Epoch {epoch+1}/{epochs} [{dt:.1f}s] "
                f"Train Loss: {train_loss:.4f}"
            )

            # Save best embedding model
            if train_loss < best_embed_train_loss:
                best_embed_train_loss = train_loss
                best_embed_model_state_dict = model.state_dict()
                ckpt_path = Path(checkpoint_dir) / f"{Path(args.config).stem}_embed.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"  --> New best embedding model (Train Loss: {best_embed_train_loss:.4f})")

    else:
        best_embed_model = torch.load((Path(checkpoint_dir) / f"{Path(args.config).stem}_embed.pt"))
        best_embed_model_state_dict = best_embed_model["model_state_dict"]

    print("Train classifier head...")
    # Load best embedding model before training classifier
    model.load_state_dict(best_embed_model_state_dict)

    for epoch in range(epochs):
        ep_start = time.time()

        # Train for one epoch
        train_loss, train_acc = train_one_epoch_classifier(model, train_loader, classify_criterion, optimizer, device)

        # Validate for one epoch
        val_loss, val_acc = validate(model, val_loader, embed_criterion, classify_criterion, device, canonical_to_crop, canonical_to_disease)

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
