"""Visualization utilities."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import torch


def colorize_depth(depth: np.ndarray, vmin=None, vmax=None, cmap='turbo') -> np.ndarray:
    """
    Colorize depth map.
    
    Args:
        depth: (H, W) depth array in meters
        vmin, vmax: depth range for colormap
        cmap: matplotlib colormap name
        
    Returns:
        (H, W, 3) RGB uint8 array
    """
    vmin = vmin or depth[depth > 0].min() if (depth > 0).any() else 0
    vmax = vmax or depth.max() if depth.max() > 0 else 10
    
    depth_normalized = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colormap = cm.get_cmap(cmap)
    colored = colormap(depth_normalized)[:, :, :3]  # Drop alpha
    return (colored * 255).astype(np.uint8)


def save_depth_panel(rgb: torch.Tensor, sparse: torch.Tensor, 
                    pred: torch.Tensor, gt: torch.Tensor,
                    save_path: str, idx: int = 0):
    """
    Save a 4-panel visualization.
    
    Args:
        rgb: (B, 3, H, W)
        sparse: (B, 1, H, W)
        pred: (B, 1, H, W)
        gt: (B, 1, H, W)
        save_path: output path
        idx: batch index to visualize
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # RGB
    rgb_np = rgb[idx].permute(1, 2, 0).cpu().numpy()
    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-8)
    axes[0].imshow(rgb_np)
    axes[0].set_title('RGB Input')
    axes[0].axis('off')
    
    # Sparse
    sparse_np = sparse[idx, 0].cpu().numpy()
    axes[1].imshow(colorize_depth(sparse_np, vmax=10))
    axes[1].set_title('Sparse Depth')
    axes[1].axis('off')
    
    # Prediction
    pred_np = pred[idx, 0].detach().cpu().numpy()
    axes[2].imshow(colorize_depth(pred_np, vmax=10))
    axes[2].set_title('Predicted Depth')
    axes[2].axis('off')
    
    # Ground truth
    gt_np = gt[idx, 0].cpu().numpy()
    axes[3].imshow(colorize_depth(gt_np, vmax=10))
    axes[3].set_title('Ground Truth')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
