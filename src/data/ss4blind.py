"""SS4Blind dataset for depth completion."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import albumentations as A


def resolve_path(path_str, root, exts):
    """Resolve file path with fuzzy matching for extensions and zero-padding."""
    p = Path(path_str)
    root = Path(root)
    tried = []
    
    # Resolve relative paths
    if not p.is_absolute():
        p = root / p
    
    # Exact match
    if p.exists():
        return p
    
    parent = p.parent
    stem = p.stem
    stem_alt = stem.lstrip("0") or stem  # Remove leading zeros
    
    # Try extension swaps (case-insensitive)
    for e in exts:
        # Original stem
        cand = parent / (stem + e)
        tried.append(str(cand))
        if cand.exists():
            return cand
        
        # Try uppercase extension
        cand_upper = parent / (stem + e.upper())
        if cand_upper.exists():
            return cand_upper
        
        # Alternative stem (no leading zeros)
        if stem_alt != stem:
            cand2 = parent / (stem_alt + e)
            tried.append(str(cand2))
            if cand2.exists():
                return cand2
            
            cand2_upper = parent / (stem_alt + e.upper())
            if cand2_upper.exists():
                return cand2_upper
    
    # Glob fallback within the same parent folder only
    if parent.exists():
        cands = list(parent.glob(f"{stem}.*")) + list(parent.glob(f"{stem_alt}.*"))
        # Filter to allowed extensions (case-insensitive)
        exts_lower = {e.lower() for e in exts}
        cands = [c for c in cands if c.suffix.lower() in exts_lower]
        
        if len(cands) == 1:
            return cands[0]
        
        if len(cands) > 1:
            # Prefer extensions in order: .png, .jpg, .jpeg, .npy
            order = [".png", ".jpg", ".jpeg", ".npy"]
            for ext in order:
                for c in cands:
                    if c.suffix.lower() == ext:
                        return c
            # If no preferred extension, return first candidate
            return cands[0]
    
    raise FileNotFoundError(
        f"Could not resolve file for '{path_str}' in root '{root}'.\n"
        f"Parent folder: {parent}\n"
        f"Tried: {'; '.join(tried[:10])}..."
    )


class SS4BlindDataset(Dataset):
    """SS4Blind depth completion dataset."""
    
    def __init__(self, csv_file: str, image_size: Tuple[int, int], 
                 sparse_points: int, is_train: bool = False, data_root: str = "."):
        """
        Args:
            csv_file: Path to CSV with columns rgb_path, depth_path, [sem_path]
            image_size: (H, W) target size
            sparse_points: Number of random points to sample for sparse depth (0 = use all valid)
            is_train: Whether to apply training augmentations
            data_root: Root directory for resolving relative paths
        """
        self.csv_file = csv_file
        self.image_size = image_size
        self.sparse_points = sparse_points
        self.is_train = is_train
        self.root = Path(data_root)
        
        # Load CSV
        self.df = pd.read_csv(csv_file)
        self.has_sem = 'sem_path' in self.df.columns
        
        # Setup augmentations
        if is_train:
            self.geo_aug = A.Compose([
                A.HorizontalFlip(p=0.5),
            ])
            self.color_aug = A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ])
        else:
            self.geo_aug = None
            self.color_aug = None
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Resolve paths with fuzzy matching
        rgb_path = resolve_path(row['rgb_path'], self.root, [".png", ".jpg", ".jpeg"])
        depth_path = resolve_path(row['depth_path'], self.root, [".npy", ".png"])
        
        # Load RGB
        rgb = np.array(Image.open(rgb_path).convert('RGB'))
        
        # Load depth (float32 meters with NaN for invalid)
        depth = np.load(depth_path).astype(np.float32)
        
        # Create valid mask
        mask = (~np.isnan(depth)).astype(np.float32)
        
        # Replace NaN with 0 for processing
        depth = np.nan_to_num(depth, nan=0.0)
        
        # Resize if needed
        if rgb.shape[:2] != self.image_size:
            rgb = cv2.resize(rgb, (self.image_size[1], self.image_size[0]), 
                           interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), 
                             interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Load semantic if available
        sem = None
        if self.has_sem and pd.notna(row.get('sem_path')) and str(row['sem_path']).strip():
            try:
                sem_path = resolve_path(row['sem_path'], self.root, [".png"])
                sem = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
            except FileNotFoundError:
                sem = None  # Semantic is optional
            if sem is not None and sem.shape[:2] != self.image_size:
                sem = cv2.resize(sem, (self.image_size[1], self.image_size[0]),
                               interpolation=cv2.INTER_NEAREST)
        
        # Apply geometric augmentations (to all modalities consistently)
        if self.geo_aug is not None:
            # Prepare for albumentations
            additional_targets = {'mask': 'mask', 'depth': 'mask'}
            if sem is not None:
                additional_targets['sem'] = 'mask'
            
            aug_compose = A.Compose([self.geo_aug], additional_targets=additional_targets)
            
            aug_input = {'image': rgb, 'mask': mask, 'depth': depth}
            if sem is not None:
                aug_input['sem'] = sem
            
            augmented = aug_compose(**aug_input)
            rgb = augmented['image']
            mask = augmented['mask']
            depth = augmented['depth']
            if sem is not None:
                sem = augmented['sem']
        
        # Apply color jitter (RGB only)
        if self.color_aug is not None:
            rgb = self.color_aug(image=rgb)['image']
        
        # Create sparse depth
        sparse_depth, sparse_mask = self._make_sparse(depth, mask, self.sparse_points)
        
        # Normalize RGB to [0, 1]
        rgb = rgb.astype(np.float32) / 255.0
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()  # (C, H, W)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()     # (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()       # (1, H, W)
        sparse_depth_tensor = torch.from_numpy(sparse_depth).unsqueeze(0).float()
        sparse_mask_tensor = torch.from_numpy(sparse_mask).unsqueeze(0).float()
        
        result = {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor,
            'sparse_depth': sparse_depth_tensor,
            'sparse_mask': sparse_mask_tensor,
        }
        
        # Add semantic if available
        if sem is not None:
            sem_tensor = torch.from_numpy(sem).unsqueeze(0).long()  # (1, H, W)
            result['sem'] = sem_tensor
        
        return result
    
    def _make_sparse(self, depth: np.ndarray, mask: np.ndarray, 
                    n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sparse depth by sampling valid pixels."""
        sparse = np.zeros_like(depth)
        sparse_mask = np.zeros_like(mask)
        
        # Get valid indices
        valid_indices = np.where(mask > 0)
        n_valid = len(valid_indices[0])
        
        if n_valid == 0:
            return sparse, sparse_mask
        
        if n_points > 0:
            # Sample n_points
            n_sample = min(n_points, n_valid)
            sample_idx = np.random.choice(n_valid, n_sample, replace=False)
            y_coords = valid_indices[0][sample_idx]
            x_coords = valid_indices[1][sample_idx]
        else:
            # Use all valid points
            y_coords = valid_indices[0]
            x_coords = valid_indices[1]
        
        sparse[y_coords, x_coords] = depth[y_coords, x_coords]
        sparse_mask[y_coords, x_coords] = 1.0
        
        return sparse, sparse_mask

