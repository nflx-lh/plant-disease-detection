"""
This module provides a factory function for loading and customizing CNN baseline models.

Supported architectures:
- MobileNetV3 Small: Optimized for speed and low-resource devices.
- EfficientNet-B0: Balanced performance focused on accuracy and efficiency.
- Vision Transformer (ViT) Base Patch16 224: Transformer-based model adapted for image classification.

The script replaces the final classification layer to match the
required number of classes for the Plant Disease Detection project (default: 26).
"""

import torch.nn as nn
import timm
import src.cct.cct as cct

# import ssl

# # Bypass SSL certificate verification for model downloading
# ssl._create_default_https_context = ssl._create_unverified_context


def get_model(model_name: str, num_classes: int = 26, pretrained: bool = True):
    """
    Entry point to get a model.
    Supported: 'mobilenet_v3_small', 'efficientnet_b0', 'vit_base_patch16_224',
    'maxvit_base_tf_224', 'cct_14_7x2_224', 'swin_base_patch4_window7_224'
    """
    weights = "DEFAULT" if pretrained else None

    if model_name == "mobilenet_v3_small":
        model = timm.create_model(
            "mobilenetv3_small_100", pretrained=pretrained, num_classes=num_classes
        )

    elif model_name == "efficientnet_b0":
        model = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=num_classes
        )

    elif model_name == "vit_base_patch16_224":
        model = timm.create_model(
            "vit_base_patch16_224",
            img_size=224,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        # Freeze ViT layers
        for param in model.blocks.parameters():
            param.requires_grad = False

        # Unfreeze classifier
        for param in model.head.parameters():
            param.requires_grad = True

        # Unfreeze last few encoder layers
        for block in model.blocks[-4:]:
            for param in block.parameters():
                param.requires_grad = True

    elif model_name == "cct_14_7x2_224":
        model = timm.create_model(
            "cct_14_7x2_224",
            img_size=224,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        for name, param in model.named_parameters():
            if any(x in name for x in ["classifier.fc", "blocks.13", "blocks.12"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif model_name == "swin_base_patch4_window7_224":
        model = timm.create_model(
            "swin_base_patch4_window7_224",
            img_size=224,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last two stages and the norm and head layers
        for stage in model.layers[-2:]:
            for param in stage.parameters():
                param.requires_grad = True

        for param in model.norm.parameters():
            param.requires_grad = True

        for param in model.head.parameters():
            param.requires_grad = True

    elif model_name == "maxvit_base_tf_224":
        model = timm.create_model(
            "maxvit_base_tf_224",
            img_size=224,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last stage and the head layers
        for stage in model.stages[-1:]:
            for param in stage.parameters():
                param.requires_grad = True

        for param in model.head.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
