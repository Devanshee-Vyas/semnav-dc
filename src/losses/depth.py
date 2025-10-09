"""Depth completion loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import sobel


class DepthLoss(nn.Module):
    """Combined depth loss with L1, gradient, and SSIM components."""
    
    def __init__(self, l1_weight=1.0, grad_weight=0.5, ssim_weight=0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor, 
                mask: torch.Tensor = None) -> dict:
        """
        Args:
            pred: (B, 1, H, W) predicted depth
            gt: (B, 1, H, W) ground truth depth
            mask: (B, 1, H, W) valid mask (optional)
            
        Returns:
            dict with total loss and components
        """
        if mask is None:
            mask = (gt > 0).float()
        
        losses = {}
        total_loss = 0.0
        
        # L1 loss
        if self.l1_weight > 0:
            l1_loss = masked_l1_loss(pred, gt, mask)
            losses['l1'] = l1_loss
            total_loss += self.l1_weight * l1_loss
        
        # Gradient loss
        if self.grad_weight > 0:
            grad_loss = gradient_loss(pred, gt, mask)
            losses['grad'] = grad_loss
            total_loss += self.grad_weight * grad_loss
        
        # SSIM loss
        if self.ssim_weight > 0:
            ssim_loss = 1.0 - ssim_loss_fn(pred, gt, mask)
            losses['ssim'] = ssim_loss
            total_loss += self.ssim_weight * ssim_loss
        
        losses['total'] = total_loss
        return losses


def masked_l1_loss(pred: torch.Tensor, gt: torch.Tensor, 
                   mask: torch.Tensor) -> torch.Tensor:
    """Masked L1 loss."""
    diff = torch.abs(pred - gt)
    masked_diff = diff * mask
    return masked_diff.sum() / (mask.sum() + 1e-8)


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor, 
                 mask: torch.Tensor) -> torch.Tensor:
    """
    Gradient loss using Sobel operator.
    Encourages sharp edges in prediction.
    """
    # Compute gradients
    pred_dx = sobel_x(pred)
    pred_dy = sobel_y(pred)
    gt_dx = sobel_x(gt)
    gt_dy = sobel_y(gt)
    
    # Erode mask slightly to avoid boundary artifacts
    mask_eroded = F.max_pool2d(mask, 3, stride=1, padding=1)
    mask_eroded = (mask_eroded == 1.0).float()
    
    # L1 on gradients
    grad_diff = torch.abs(pred_dx - gt_dx) + torch.abs(pred_dy - gt_dy)
    masked_grad = grad_diff * mask_eroded
    
    return masked_grad.sum() / (mask_eroded.sum() + 1e-8)


def sobel_x(tensor: torch.Tensor) -> torch.Tensor:
    """Apply Sobel X filter."""
    kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
    return F.conv2d(tensor, kernel, padding=1)


def sobel_y(tensor: torch.Tensor) -> torch.Tensor:
    """Apply Sobel Y filter."""
    kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
    return F.conv2d(tensor, kernel, padding=1)


def ssim_loss_fn(pred: torch.Tensor, gt: torch.Tensor, 
                 mask: torch.Tensor, window_size=11) -> torch.Tensor:
    """
    SSIM on normalized inverse depth.
    """
    # Convert to inverse depth
    pred_inv = 1.0 / (pred + 1e-3)
    gt_inv = 1.0 / (gt + 1e-3)
    
    # Normalize to [0, 1]
    pred_inv = (pred_inv - pred_inv.min()) / (pred_inv.max() - pred_inv.min() + 1e-8)
    gt_inv = (gt_inv - gt_inv.min()) / (gt_inv.max() - gt_inv.min() + 1e-8)
    
    # Simple SSIM approximation (luminance only for speed)
    C1 = 0.01 ** 2
    
    mu_pred = F.avg_pool2d(pred_inv, window_size, stride=1, padding=window_size//2)
    mu_gt = F.avg_pool2d(gt_inv, window_size, stride=1, padding=window_size//2)
    
    mu_pred_sq = mu_pred ** 2
    mu_gt_sq = mu_gt ** 2
    mu_pred_gt = mu_pred * mu_gt
    
    sigma_pred_sq = F.avg_pool2d(pred_inv ** 2, window_size, stride=1, 
                                 padding=window_size//2) - mu_pred_sq
    sigma_gt_sq = F.avg_pool2d(gt_inv ** 2, window_size, stride=1, 
                               padding=window_size//2) - mu_gt_sq
    sigma_pred_gt = F.avg_pool2d(pred_inv * gt_inv, window_size, stride=1, 
                                 padding=window_size//2) - mu_pred_gt
    
    ssim_map = ((2 * mu_pred_gt + C1) * (2 * sigma_pred_gt + C1)) / \
               ((mu_pred_sq + mu_gt_sq + C1) * (sigma_pred_sq + sigma_gt_sq + C1))
    
    # Mask and average
    mask_pooled = F.avg_pool2d(mask, window_size, stride=1, padding=window_size//2)
    mask_valid = (mask_pooled > 0.5).float()
    
    ssim_masked = ssim_map * mask_valid
    return ssim_masked.sum() / (mask_valid.sum() + 1e-8)
