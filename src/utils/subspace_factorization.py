import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import random
from torch.utils.data import Dataset
import torchvision.transforms as T

from .baseline_models import get_model
from .transformations import _normalize_steps, _build_transform_from_step


class SubspaceDGModel(nn.Module):
    def __init__(self, backbone_name="vit_base_patch16_224", embed_dim=256,
                 num_classes=26, num_styles=4, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = get_model(backbone_name, num_classes=num_classes, pretrained=pretrained, unfreeze_backbone=False)
        feat_dim = self.backbone.num_features
        embed_dim = feat_dim

        for param in self.backbone.parameters():
            param.requires_grad = False

        # self.adapter = nn.Linear(feat_dim, embed_dim)
        self.class_subspace = nn.Linear(embed_dim, embed_dim, bias=False)
        self.style_subspace = nn.Linear(embed_dim, embed_dim, bias=False)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.style_classifier = nn.Linear(embed_dim, num_styles)

    def forward(self, x):
        feat = self.backbone.forward_features(x)

        if self.backbone_name in ["efficientnet_b0", "mobilenet_v3_small"]:
            feat = nn.functional.adaptive_avg_pool2d(feat, 1)
            feat = feat.flatten(1)
        else:
            feat = self.backbone.forward_head(feat, pre_logits=True)

        # z = self.adapter(feat)
        z = nn.functional.normalize(feat, dim=1)
        z_class = nn.functional.normalize(self.class_subspace(z), dim=1)
        z_style = nn.functional.normalize(self.style_subspace(z), dim=1)
        class_logits = self.classifier(z_class)
        style_logits = self.style_classifier(z_style)
        return z, z_class, z_style, class_logits, style_logits
    
class DGTwoViewWrapper(Dataset):
    def __init__(self, base_dataset, timm_transform, img_size, transforms_config):
        self.base_dataset = base_dataset
        self.timm_transform = timm_transform
        self.img_size = img_size
        
        augmentation_choices = _normalize_steps(transforms_config)
        self.augmentations = [
            op for s in augmentation_choices if (op := _build_transform_from_step(s)) is not None
        ]

        self.n_transforms = len(self.augmentations)

        if self.n_transforms == 0:
            raise ValueError("You should pass a list of possible transforms for DGSubspaceFactorization.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]

        # Clean view
        img_clean = self.timm_transform(image)

        # Augmented view
        aug_id = random.randint(0, self.n_transforms-1)
        aug = self.augmentations[aug_id]
        img_aug = self.timm_transform(aug(image)) if not isinstance(aug, T.RandomErasing) else self.timm_transform(image)

        if isinstance(aug, T.RandomErasing):
            img_aug = aug(img_aug)

        return img_clean, img_aug, label, aug_id
