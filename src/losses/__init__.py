"""Loss functions for depth completion."""
from .depth import DepthLoss
from .boundary import BoundaryIoULoss

__all__ = ['DepthLoss', 'BoundaryIoULoss']
