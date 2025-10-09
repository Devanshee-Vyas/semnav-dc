"""Boundary-aware losses for depth completion."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryIoULoss(nn.Module):
    """
    Boundary IoU loss.
    Focuses on 3-pixel bands around GT depth edges.
    """
    
    def __init__(self, band_width=3):
        super().__init__()
        self.band_width = band_width
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) predicted depth
            gt: (B, 1, H, W) ground truth depth
            mask: (B, 1, H, W) valid mask
            
        Returns:
            Boundary IoU loss (lower is better)
        """
        if mask is None:
            mask = (gt > 0).float()
        
        # Compute GT edges using Sobel
        gt_dx = sobel_x(gt)
        gt_dy = sobel_y(gt)
        gt_edges = torch.sqrt(gt_dx**2 + gt_dy**2)
        gt_edges = (gt_edges > 0.1).float()  # Threshold
        
        # Dilate edges to create band
        boundary_mask = dilate(gt_edges, self.band_width)
        boundary_mask = boundary_mask * mask
        
        # Compute error in boundary regions
        error = torch.abs(pred - gt)
        
        # Normalize error to [0, 1] for IoU computation
        error_norm = error / (gt + 1e-3)
        error_binary = (error_norm < 0.1).float()  # Threshold: 10% relative error
        
        # IoU in boundary regions
        intersection = (error_binary * boundary_mask).sum()
        union = boundary_mask.sum()
        
        boundary_iou = intersection / (union + 1e-8)
        
        # Return as loss (1 - IoU)
        return 1.0 - boundary_iou


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


def dilate(tensor: torch.Tensor, iterations: int) -> torch.Tensor:
    """Binary dilation using max pooling."""
    kernel_size = 2 * iterations + 1
    dilated = F.max_pool2d(tensor, kernel_size, stride=1, padding=iterations)
    return dilated
