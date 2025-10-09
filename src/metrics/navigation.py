"""Navigation-specific metrics for blind assistance."""
import torch
import numpy as np


def compute_navigation_metrics(pred_depth: torch.Tensor, gt_depth: torch.Tensor,
                               camera_height: float = 1.3,
                               obstacle_threshold: float = 2.0) -> dict:
    """
    Compute navigation metrics.
    
    Args:
        pred_depth: (B, 1, H, W) predicted depth
        gt_depth: (B, 1, H, W) ground truth depth
        camera_height: camera height in meters (default 1.3m for handheld)
        obstacle_threshold: distance threshold for obstacles (meters)
        
    Returns:
        dict with free_space_iou and obstacle_recall_2m
    """
    # Derive free-space masks using camera height heuristic
    # Simple rule: bottom 1/3 of image with depth > 1.0m is likely free space
    B, _, H, W = pred_depth.shape
    
    # Ground plane mask (bottom half of image)
    ground_mask = torch.zeros_like(pred_depth)
    ground_mask[:, :, H//2:, :] = 1.0
    
    # Free space: ground region with depth > 1.0m
    pred_freespace = (pred_depth > 1.0).float() * ground_mask
    gt_freespace = (gt_depth > 1.0).float() * ground_mask
    
    # Free space IoU
    intersection = (pred_freespace * gt_freespace).sum()
    union = ((pred_freespace + gt_freespace) > 0).float().sum()
    freespace_iou = (intersection / (union + 1e-8)).item()
    
    # Obstacle recall at 2m
    # Obstacles: depth < threshold in upper half of image
    obstacle_mask = torch.zeros_like(pred_depth)
    obstacle_mask[:, :, :H//2, :] = 1.0
    
    gt_obstacles = ((gt_depth < obstacle_threshold) & (gt_depth > 0)).float() * obstacle_mask
    pred_obstacles = (pred_depth < obstacle_threshold).float() * obstacle_mask
    
    # Recall: TP / (TP + FN)
    true_positives = (pred_obstacles * gt_obstacles).sum()
    obstacle_recall = (true_positives / (gt_obstacles.sum() + 1e-8)).item()
    
    return {
        'freespace_iou': freespace_iou,
        'obstacle_recall_2m': obstacle_recall
    }


def batch_compute_navigation_metrics(pred_depth: torch.Tensor, 
                                     gt_depth: torch.Tensor) -> dict:
    """
    Compute navigation metrics over a batch and average.
    
    Args:
        pred_depth: (B, 1, H, W)
        gt_depth: (B, 1, H, W)
        
    Returns:
        dict with averaged navigation metrics
    """
    metrics_list = []
    B = pred_depth.shape[0]
    
    for i in range(B):
        metrics = compute_navigation_metrics(
            pred_depth[i:i+1], 
            gt_depth[i:i+1]
        )
        metrics_list.append(metrics)
    
    # Average
    avg_metrics = {
        k: np.mean([m[k] for m in metrics_list])
        for k in metrics_list[0].keys()
    }
    
    return avg_metrics
