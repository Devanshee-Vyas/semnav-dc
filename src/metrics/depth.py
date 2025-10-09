"""Depth evaluation metrics."""
import torch
import numpy as np


def compute_depth_metrics(pred: torch.Tensor, gt: torch.Tensor, 
                         mask: torch.Tensor = None) -> dict:
    """
    Compute standard depth metrics.
    
    Args:
        pred: (B, 1, H, W) predicted depth
        gt: (B, 1, H, W) ground truth depth
        mask: (B, 1, H, W) valid mask (optional)
        
    Returns:
        dict with RMSE, MAE, iRMSE, iMAE
    """
    if mask is None:
        mask = (gt > 0).float()
    
    # Flatten
    pred_valid = pred[mask > 0]
    gt_valid = gt[mask > 0]
    
    if len(pred_valid) == 0:
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'irmse': float('nan'),
            'imae': float('nan')
        }
    
    # RMSE
    rmse = torch.sqrt(torch.mean((pred_valid - gt_valid) ** 2))
    
    # MAE
    mae = torch.mean(torch.abs(pred_valid - gt_valid))
    
    # Inverse RMSE
    pred_inv = 1.0 / (pred_valid + 1e-3)
    gt_inv = 1.0 / (gt_valid + 1e-3)
    irmse = torch.sqrt(torch.mean((pred_inv - gt_inv) ** 2))
    
    # Inverse MAE
    imae = torch.mean(torch.abs(pred_inv - gt_inv))
    
    return {
        'rmse': rmse.item(),
        'mae': mae.item(),
        'irmse': irmse.item(),
        'imae': imae.item()
    }


def batch_compute_metrics(pred: torch.Tensor, gt: torch.Tensor, 
                         mask: torch.Tensor = None) -> dict:
    """
    Compute metrics over a batch and average.
    
    Args:
        pred: (B, 1, H, W)
        gt: (B, 1, H, W)
        mask: (B, 1, H, W)
        
    Returns:
        dict with averaged metrics
    """
    metrics_list = []
    B = pred.shape[0]
    
    for i in range(B):
        m = mask[i:i+1] if mask is not None else None
        metrics = compute_depth_metrics(pred[i:i+1], gt[i:i+1], m)
        if not np.isnan(metrics['rmse']):
            metrics_list.append(metrics)
    
    if len(metrics_list) == 0:
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'irmse': float('nan'),
            'imae': float('nan')
        }
    
    # Average
    avg_metrics = {
        k: np.mean([m[k] for m in metrics_list])
        for k in metrics_list[0].keys()
    }
    
    return avg_metrics
