"""Data augmentation transforms."""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_train_transforms(image_size=(240, 320)):
    """
    Get training transforms with albumentations.
    Geometric transforms apply to all inputs; color only to RGB.
    """
    return A.Compose([
        # Geometric transforms (apply to all)
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        A.Resize(height=image_size[0], width=image_size[1]),
    ], additional_targets={
        'sparse': 'mask',
        'mask': 'mask',
        'gt': 'mask'
    })


def get_color_transforms():
    """Color transforms for RGB only."""
    return A.Compose([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    ])


def get_val_transforms(image_size=(240, 320)):
    """Get validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
    ], additional_targets={
        'sparse': 'mask',
        'mask': 'mask',
        'gt': 'mask'
    })


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Normalize RGB to ImageNet stats.
    
    Args:
        rgb: (H, W, 3) uint8 image
        
    Returns:
        (3, H, W) float32 tensor, normalized
    """
    rgb = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean) / std
    return rgb.transpose(2, 0, 1)  # HWC -> CHW
