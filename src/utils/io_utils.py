"""I/O utilities for loading and saving data."""
import os
import csv
from pathlib import Path
from typing import Dict, Any
import yaml
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, metrics: Dict[str, float], save_path: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, save_path)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer = None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def write_csv_log(log_path: str, data: Dict[str, Any], mode='a'):
    """Append metrics to CSV log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)
    
    with open(log_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists or mode == 'w':
            writer.writeheader()
        writer.writerow(data)
