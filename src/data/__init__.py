"""Data loading and preprocessing."""
from pathlib import Path
from .ss4blind import SS4BlindDataset
from .transforms import get_train_transforms, get_val_transforms

__all__ = ['SS4BlindDataset', 'get_train_transforms', 'get_val_transforms', 'make_dataset']


def make_dataset(dataset_name: str, split: str, cfg: dict):
    """Factory function to create dataset based on config."""
    if dataset_name == 'ss4blind':
        data_root = cfg.get('data_root', 'data/ss4blind')
        splits_dir = Path(cfg.get('splits_dir', Path(data_root) / 'splits'))
        csv_file = cfg.get(f'{split}_csv', f'{split}.csv')
        csv_path = splits_dir / csv_file
        
        image_size = tuple(cfg.get('image_size', [480, 640]))
        sparse_points = cfg.get('sparse_points', 800)
        is_train = (split == 'train')
        
        return SS4BlindDataset(str(csv_path), image_size, sparse_points, is_train, data_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
