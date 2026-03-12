# src/train/train_dg.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import sys
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import timm

from timm.data import resolve_model_data_config, create_transform

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.dataloaders import PlantDiseaseDataset, get_val_dataloader
from src.utils.subspace_factorization import DGTwoViewWrapper, SubspaceDGModel
from src.utils.supcon import SupConLoss

from torch.utils.data import DataLoader

import json



# -----------------------------
# Training loop
# -----------------------------
def train_one_epoch(model, loader, supcon_criterion, ce_criterion, optimizer, device, lambda_orth=0.01):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for img_clean, img_aug, labels, aug_ids in pbar:
        labels = labels.to(device)
        aug_ids = aug_ids.to(device)
        images = torch.cat([img_clean, img_aug], dim=0).to(device)

        optimizer.zero_grad()

        z, z_class, z_style, class_logits, style_logits = model(images)
        batch_size = labels.size(0)
        # Prepare features for SupCon
        zc1, zc2 = torch.split(z_class, batch_size)
        features_class = torch.stack([zc1, zc2], dim=1)
        zs1, zs2 = torch.split(z_style, batch_size)
        features_style = torch.stack([zs1, zs2], dim=1)

        loss_supcon_class = supcon_criterion(features_class, labels)
        loss_supcon_style = supcon_criterion(features_style, aug_ids)
        loss_class = ce_criterion(class_logits[:batch_size], labels)
        loss_style = ce_criterion(style_logits[:batch_size], aug_ids)
        # Orthogonality
        Wc, Ws = model.class_subspace.weight, model.style_subspace.weight
        loss_orth = torch.norm(Wc @ Ws.T, p='fro')

        loss = loss_supcon_class + 0.5 * loss_supcon_style + loss_class + 0.25 * loss_style + lambda_orth * loss_orth
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        _, predicted = class_logits[:batch_size].max(1)
        correct += predicted.eq(labels).sum().item()
        total += batch_size
        pbar.set_postfix(loss=loss.item(), acc=correct/total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# -----------------------------
# Validation
# -----------------------------
def validate(model, loader, ce_criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            _, _, _, class_logits, _ = model(images)
            loss = ce_criterion(class_logits, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = class_logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss/total, correct/total

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", ".")
    splits_dir = config.get("splits_dir", "data/splits")
    output_dir = Path(config.get("output_dir","outputs"))
    checkpoint_dir = Path(config.get("checkpoint_dir","checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Deserialize transformations config
    transform_config = config.get("transformations", None)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Dataset
    train_csv = Path(splits_dir)/"pv_train.csv"
    val_csv = Path(splits_dir)/"pv_val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Run M1 pipeline first")

    base_train_dataset = PlantDiseaseDataset(str(train_csv), root_dir=data_dir, transform=None)

    # timm transforms
    model_name = config["model_name"]
    timm_model_name = "mobilenetv3_small_100" if model_name == "mobilenet_v3_small" else model_name
    with torch.no_grad():
        backbone = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
        data_config = resolve_model_data_config(backbone)
        timm_transform = create_transform(**data_config, is_training=False)
        img_size = data_config["input_size"][1]

        # Wrap dataset for SupCon
        train_dataset = DGTwoViewWrapper(base_train_dataset, timm_transform, img_size, transform_config)

        train_loader = DataLoader(train_dataset, batch_size=config["hyperparameters"].get("batch_size",32),
                                shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = get_val_dataloader(val_csv, root_dir=data_dir, batch_size=config["hyperparameters"].get("batch_size",32), transforms=timm_transform)

    # Model
    num_classes = config.get("num_classes",26)
    num_styles = train_dataset.n_transforms
    print("Number of classes: ", num_classes)
    print("Number of augmentations: ", num_styles)
    model = SubspaceDGModel(model_name, num_classes=num_classes, num_styles=num_styles)
    model.to(device)

    # Optimizer
    lr = config["hyperparameters"].get("learning_rate",1e-3)
    weight_decay = config["hyperparameters"].get("weight_decay",0.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss
    supcon_criterion = SupConLoss(temperature=0.07)
    ce_criterion = nn.CrossEntropyLoss()

    # Scheduler
    epochs = config["hyperparameters"].get("num_epochs",10)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_val_acc = 0.0
    history = []
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, supcon_criterion, ce_criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, ce_criterion, device)
        scheduler.step()

        print(f"[{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        history.append({"epoch":epoch+1, "train_loss":train_loss, "train_acc":train_acc,
                        "val_loss":val_loss, "val_acc":val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = checkpoint_dir/f"{Path(args.config).stem}_best.pt"
            torch.save({"epoch":epoch, "model_state_dict":model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),
                        "val_acc":val_acc,
                        "config":config}, ckpt_path)
            print(f"Saved new best model ({best_val_acc:.4f})")

    pd.DataFrame(history).to_csv(output_dir/f"training_log_{Path(args.config).stem}.csv", index=False)
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")

if __name__=="__main__":
    main()