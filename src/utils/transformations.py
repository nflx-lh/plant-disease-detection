"""
This module provides data transformation pipelines for different model architectures.
M4: Build transform pipeline from config.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from torchvision import transforms
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import default_collate
from torchvision.transforms.v2 import CutMix, MixUp, RandomChoice


# -----------------------------
# Helpers
# -----------------------------
def _to_tuple2(x, name: str):
    """
    Convert list/tuple of length 2 to tuple(float, float).
    """
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return (float(x[0]), float(x[1]))
    raise ValueError(f"{name} must be a list/tuple of length 2, got: {x}")


def _normalize_steps(
    transforms_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """
    Support BOTH formats:
      A) {"steps": [ ... ]}
      B) [ ... ]
    """
    if transforms_config is None:
        return []

    if isinstance(transforms_config, list):
        return transforms_config

    if isinstance(transforms_config, dict):
        steps = transforms_config.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("transforms_config['steps'] must be a list.")
        return steps

    raise ValueError("transforms_config must be dict, list, or None.")


def _build_transform_from_step(step: Dict[str, Any]):
    """
    Map config step -> torchvision transform instance.
    Supported names:
      - color_jitter
      - random_rotation
      - random_horizontal_flip
      - random_resized_crop
      - random_affine
      - gaussian_blur
      - random_perspective
      - random_erasing
      - cutmix (handled in collate_fn, returns None)
      - mixup (handled in collate_fn, returns None)
    """
    if "name" not in step:
        raise ValueError(f"Each transform step must contain 'name'. Got: {step}")

    name = step["name"]
    params = step.get("params", {}) or {}

    if name == "color_jitter":
        return transforms.ColorJitter(
            brightness=params.get("brightness", 0),
            contrast=params.get("contrast", 0),
            saturation=params.get("saturation", 0),
            hue=params.get("hue", 0),
        )

    elif name == "random_rotation":
        degrees = params.get("degrees", 0)
        return transforms.RandomRotation(degrees)

    elif name == "random_horizontal_flip":
        p = float(params.get("p", 0.5))
        return transforms.RandomHorizontalFlip(p=p)

    elif name == "random_resized_crop":
        size = params.get("size", 224)
        scale = _to_tuple2(params.get("scale", [0.8, 1.0]), "random_resized_crop.scale")
        ratio = _to_tuple2(
            params.get("ratio", [0.75, 1.3333]), "random_resized_crop.ratio"
        )
        interpolation = params.get(
            "interpolation", transforms.InterpolationMode.BILINEAR
        )
        antialias = params.get("antialias", True)
        return transforms.RandomResizedCrop(
            size=size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
            antialias=antialias,
        )

    elif name == "random_affine":
        degrees = params.get("degrees", 0)
        translate = params.get("translate", None)
        scale = params.get("scale", None)
        shear = params.get("shear", None)

        if translate is not None:
            translate = _to_tuple2(translate, "random_affine.translate")
        if scale is not None:
            scale = _to_tuple2(scale, "random_affine.scale")
        # shear can be number or sequence in torchvision

        interpolation = params.get(
            "interpolation", transforms.InterpolationMode.BILINEAR
        )
        fill = params.get("fill", 0)

        return transforms.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=fill,
        )

    elif name == "gaussian_blur":
        kernel_size = params.get("kernel_size", 3)
        sigma = params.get("sigma", [0.1, 1.0])
        if isinstance(sigma, (list, tuple)):
            sigma = _to_tuple2(sigma, "gaussian_blur.sigma")
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    elif name == "random_perspective":
        distortion_scale = float(params.get("distortion_scale", 0.1))
        p = float(params.get("p", 0.3))
        interpolation = params.get(
            "interpolation", transforms.InterpolationMode.BILINEAR
        )
        fill = params.get("fill", 0)
        return transforms.RandomPerspective(
            distortion_scale=distortion_scale,
            p=p,
            interpolation=interpolation,
            fill=fill,
        )

    elif name == "random_erasing":
        # NOTE: should be after ToTensor/Normalize stage.
        p = float(params.get("p", 0.25))
        scale = _to_tuple2(params.get("scale", [0.02, 0.1]), "random_erasing.scale")
        ratio = _to_tuple2(params.get("ratio", [0.3, 3.3]), "random_erasing.ratio")
        value = params.get("value", 0)
        inplace = bool(params.get("inplace", False))
        return transforms.RandomErasing(
            p=p,
            scale=scale,
            ratio=ratio,
            value=value,
            inplace=inplace,
        )
    
    elif name == "cutmix" or name == "mixup":
        # These are handled in collate_fn, not as part of transform pipeline
        return None

    else:
        raise ValueError(
            f"Unsupported transform name: {name}. "
            f"Supported: color_jitter, random_rotation, random_horizontal_flip, "
            f"random_resized_crop, random_affine, gaussian_blur, random_perspective, random_erasing, cutmix, mixup."
        )


def _inject_custom_train_transforms(
    train_transform, custom_steps: List[Dict[str, Any]]
):
    """
    Inject custom transforms into train pipeline only.
    - image-space transforms are inserted BEFORE tensor conversion step
      (ToTensor / PILToTensor / MaybeToTensor)
    - random_erasing is inserted AFTER normalization (or at end if normalize not found)
    """
    if not custom_steps:
        return train_transform

    custom_ops = [_build_transform_from_step(s) for s in custom_steps]

    image_ops = [
        op for op in custom_ops if not isinstance(op, transforms.RandomErasing)
    ]
    erasing_ops = [op for op in custom_ops if isinstance(op, transforms.RandomErasing)]

    if not hasattr(train_transform, "transforms"):
        return transforms.Compose(image_ops + [train_transform] + erasing_ops)

    base_ops = list(deepcopy(train_transform.transforms))

    # Helper: match by class name to support timm wrappers like MaybeToTensor
    def _cls_name(op):
        return op.__class__.__name__.lower()

    tensor_like_names = {"totensor", "piltotensor", "maybetotensor"}
    normalize_names = {"normalize"}

    # 1) Find tensor conversion position
    tensor_idx = None
    for i, op in enumerate(base_ops):
        if _cls_name(op) in tensor_like_names:
            tensor_idx = i
            break

    # Insert image-space ops before tensor conversion if found; else prepend
    if tensor_idx is None:
        ops_after_image_insert = image_ops + base_ops
    else:
        ops_after_image_insert = (
            base_ops[:tensor_idx] + image_ops + base_ops[tensor_idx:]
        )

    # 2) Place RandomErasing after normalize if normalize exists, else append
    if erasing_ops:
        norm_idx = None
        for i, op in enumerate(ops_after_image_insert):
            if _cls_name(op) in normalize_names:
                norm_idx = i
                break

        if norm_idx is None:
            final_ops = ops_after_image_insert + erasing_ops
        else:
            final_ops = (
                ops_after_image_insert[: norm_idx + 1]
                + erasing_ops
                + ops_after_image_insert[norm_idx + 1 :]
            )
    else:
        final_ops = ops_after_image_insert

    return transforms.Compose(final_ops)


# -----------------------------
# Public API
# -----------------------------
def get_transforms(
    model,
    model_name: str,
    image_size: int = 224,
    transforms_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
):
    """
    Returns appropriate data transformations for the given model.

    Args:
        model: The model object (used for ViT models to get data config)
        model_name: model name
        image_size: target image size
        transforms_config: optional config for custom training augmentations
    Returns:
        (train_transform, val_transform, test_transform)
    """
    train_transform, val_transform, test_transform = get_default_transforms(
        model, model_name, image_size
    )

    custom_steps = _normalize_steps(transforms_config)
    if custom_steps:
        # Apply custom augmentation to TRAIN only
        train_transform = _inject_custom_train_transforms(train_transform, custom_steps)

    return train_transform, val_transform, test_transform

def get_collate_fn(transforms_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None, num_classes = 26):
    """
    Returns a custom collate function if CutMix or MixUp is specified in transforms_config.

    Args:
        num_classes: Number of classes in the dataset
        transforms_config: Optional dictionary to customize transformations
    Returns:
        A collate function or None.
    """
    use_cutmix = False
    use_mixup = False
    cutmix_alpha = 1.0
    mixup_alpha = 1.0

    transforms_config = _normalize_steps(transforms_config)
    
    for step in transforms_config:
        if step["name"] == "cutmix":
            cutmix_alpha = step["params"].get("alpha", 1.0)
            cutmix_p = step["params"].get("p", 0.5)
            use_cutmix = True
        elif step["name"] == "mixup":
            mixup_alpha = step["params"].get("alpha", 1.0)
            mixup_p = step["params"].get("p", 0.5)
            use_mixup = True

    def collate_fn(batch):
        images, targets = default_collate(batch)
        fn = None
        if use_cutmix and use_mixup:
            fn = RandomChoice([CutMix(alpha=cutmix_alpha, num_classes=num_classes),
                                MixUp(alpha=mixup_alpha, num_classes=num_classes)], 
                              p=[cutmix_p, mixup_p])
        elif use_cutmix:
            fn = CutMix(alpha=cutmix_alpha, num_classes=num_classes)
        elif use_mixup:
            fn = MixUp(alpha=mixup_alpha, num_classes=num_classes)

        if fn is not None:
            images, targets = fn(images, targets)
        return images, targets

    return collate_fn

def get_default_transforms(model, model_name: str, image_size: int = 224):
    """
    Returns default data transformations for the given model.
    """

    if model_name in [
        "mobilenet_v3_small",
        "efficientnet_b0",
        "vit_base_patch16_224",
        "swin_base_patch4_window7_224",
        "maxvit_base_tf_224",
        "cct_14_7x2_224",
    ]:
        data_config = resolve_model_data_config(model)

        train_transform = create_transform(**data_config, is_training=True)

        val_transform = create_transform(**data_config, is_training=False)

        test_transform = create_transform(**data_config, is_training=False)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return train_transform, val_transform, test_transform
