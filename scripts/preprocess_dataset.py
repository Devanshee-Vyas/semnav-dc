"""Ingest SS4Blind dataset into canonical format."""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def read_pfm(file_path: Path) -> np.ndarray:
    """Read PFM (Portable Float Map) file."""
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header not in ('PF', 'Pf'):
            raise ValueError(f'Not a PFM file: {file_path}')
        
        # Read dimensions
        dims = f.readline().decode('utf-8').rstrip()
        width, height = map(int, dims.split())
        
        # Read scale/endianness
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)
        
        # Read data
        data = np.fromfile(f, endian + 'f')
        
    # Reshape (height, width) for grayscale or (height, width, 3) for color
    if header == 'PF':  # Color
        data = data.reshape((height, width, 3))
    else:  # Grayscale
        data = data.reshape((height, width))
    
    # Flip vertically (PFM stores bottom-to-top)
    data = np.flipud(data)
    
    return data * scale


def read_depth_any(path: Path) -> np.ndarray:
    """Read depth from various formats, return float32 meters with NaN for invalid."""
    ext = path.suffix.lower()
    
    if ext == '.npy':
        depth = np.load(path).astype(np.float32)
    elif ext == '.pfm':
        depth = read_pfm(path).astype(np.float32)
    elif ext == '.png':
        # Assume uint16 millimeters
        depth_uint = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if depth_uint is None:
            raise ValueError(f"Could not read depth PNG: {path}")
        depth = depth_uint.astype(np.float32) / 1000.0  # mm to meters
    else:
        raise ValueError(f"Unsupported depth format: {ext}")
    
    # Set non-positive values to NaN
    depth[depth <= 0] = np.nan
    
    return depth


def find_files(src_dir: Path) -> Dict[str, List[Path]]:
    """Find all RGB, depth, and semantic files."""
    files = {'rgb': [], 'depth': [], 'sem': []}
    
    # RGB extensions
    rgb_exts = {'.png', '.jpg', '.jpeg'}
    # Depth extensions
    depth_exts = {'.png', '.npy', '.pfm'}
    # Semantic extensions
    sem_exts = {'.png'}
    
    for f in src_dir.rglob('*'):
        if not f.is_file():
            continue
        ext_lower = f.suffix.lower()
        
        # Check both filename AND parent folder path for classification
        name_lower = f.stem.lower()
        parent_path_lower = str(f.parent).lower()
        
        # Depth files: check name or if in 'depth' folder
        if 'depth' in name_lower or 'disp' in name_lower or 'depth' in parent_path_lower:
            if ext_lower in depth_exts:
                files['depth'].append(f)
        # Semantic files
        elif 'sem' in name_lower or 'label' in name_lower or 'semantic' in parent_path_lower:
            if ext_lower in sem_exts:
                files['sem'].append(f)
        # RGB files: check name or if in 'rgb' or 'color' folder
        elif 'color' in name_lower or 'rgb' in name_lower or 'rgb' in parent_path_lower or 'color' in parent_path_lower:
            if ext_lower in rgb_exts:
                files['rgb'].append(f)
        # Default: treat as RGB if common image extension
        elif ext_lower in rgb_exts:
            files['rgb'].append(f)
    
    return files


def pair_samples(files: Dict[str, List[Path]]) -> List[Dict[str, Path]]:
    """Pair RGB, depth, and optionally semantic files by basename."""
    # Build dicts keyed by base ID (strip suffixes like -color, -depth, etc.)
    def get_base_id(path: Path) -> str:
        """Extract base ID from filename like 0016-color.png -> 0016"""
        stem = path.stem
        # Remove common suffixes
        for suffix in ['-color', '-depth', '-rgb', '-disp', '-sem', '-label']:
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
        return stem
    
    rgb_dict = {get_base_id(f): f for f in files['rgb']}
    depth_dict = {get_base_id(f): f for f in files['depth']}
    sem_dict = {get_base_id(f): f for f in files['sem']}
    
    samples = []
    for base_id, rgb_path in rgb_dict.items():
        # Must have depth
        if base_id not in depth_dict:
            continue
        
        sample = {
            'rgb': rgb_path,
            'depth': depth_dict[base_id]
        }
        
        # Optional semantic
        if base_id in sem_dict:
            sample['sem'] = sem_dict[base_id]
        
        samples.append(sample)
    
    return samples


def get_scene_id(file_path: Path, src_dir: Path) -> str:
    """Extract scene ID from file path."""
    rel_path = file_path.relative_to(src_dir)
    # Use parent folder (e.g., realsensed435) as scene
    if len(rel_path.parts) >= 3:
        return rel_path.parts[-2]  # Parent folder of file
    elif len(rel_path.parts) >= 2:
        return rel_path.parts[-2]
    return "default_scene"


def split_samples(samples: List[Dict[str, Path]], src_dir: Path, 
                 val_pct: float, test_pct: float, seed: int) -> Tuple[List, List, List]:
    """Split samples by scene to avoid leakage."""
    # Group by scene
    scene_to_samples = {}
    for sample in samples:
        scene = get_scene_id(sample['rgb'], src_dir)
        scene_to_samples.setdefault(scene, []).append(sample)
    
    # Sort scenes for determinism
    scenes = sorted(scene_to_samples.keys())
    
    # Shuffle with seed
    np.random.seed(seed)
    np.random.shuffle(scenes)
    
    # Calculate split sizes
    n_scenes = len(scenes)
    n_test = max(1, int(n_scenes * test_pct))
    n_val = max(1, int(n_scenes * val_pct))
    n_train = n_scenes - n_test - n_val
    
    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train:n_train + n_val]
    test_scenes = scenes[n_train + n_val:]
    
    # Collect samples
    train = [s for sc in train_scenes for s in scene_to_samples[sc]]
    val = [s for sc in val_scenes for s in scene_to_samples[sc]]
    test = [s for sc in test_scenes for s in scene_to_samples[sc]]
    
    return train, val, test


def resize_safe(img: np.ndarray, target_hw: Tuple[int, int], 
               mode: str = 'bilinear') -> np.ndarray:
    """Resize image/depth safely."""
    h, w = target_hw
    
    if mode == 'bilinear':
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    elif mode == 'nearest':
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def write_sample(sample: Dict[str, Path], dst_dir: Path, split: str, 
                stem: str, target_hw: Tuple[int, int]) -> Dict[str, str]:
    """Process and write one sample."""
    # Read RGB
    rgb = cv2.imread(str(sample['rgb']))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = resize_safe(rgb, target_hw, 'bilinear')
    
    # Read depth
    depth = read_depth_any(sample['depth'])
    depth = resize_safe(depth, target_hw, 'nearest')
    
    # Read semantic if available
    sem = None
    if 'sem' in sample:
        sem = cv2.imread(str(sample['sem']), cv2.IMREAD_GRAYSCALE)
        if sem is not None:
            sem = resize_safe(sem, target_hw, 'nearest')
    
    # Save RGB
    rgb_out = dst_dir / 'rgb' / split / f'{stem}.png'
    rgb_out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(rgb_out)
    
    # Save depth
    depth_out = dst_dir / 'depth' / split / f'{stem}.npy'
    depth_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(depth_out, depth)
    
    # Save semantic if available
    sem_out = None
    if sem is not None:
        sem_out = dst_dir / 'sem' / split / f'{stem}.png'
        sem_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sem_out), sem)
    
    # Return relative paths (relative to dst_dir, not its parent)
    result = {
        'rgb_path': str(rgb_out.relative_to(dst_dir)),
        'depth_path': str(depth_out.relative_to(dst_dir))
    }
    if sem_out:
        result['sem_path'] = str(sem_out.relative_to(dst_dir))
    
    return result


def write_split_csv(rows: List[Dict[str, str]], csv_path: Path):
    """Write CSV for a split."""
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def create_qc_plot(dst_dir: Path, split_data_dict: Dict[str, List[Dict]]):
    """Create QC plot with depth statistics and example."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Load first train sample for example
    if split_data_dict['train']:
        example = split_data_dict['train'][0]
        rgb = np.array(Image.open(dst_dir / example['rgb_path']))
        depth = np.load(dst_dir / example['depth_path'])
        
        # RGB
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('Example RGB')
        axes[0, 0].axis('off')
        
        # Valid mask
        valid_mask = ~np.isnan(depth)
        axes[0, 1].imshow(valid_mask, cmap='gray')
        axes[0, 1].set_title(f'Valid Mask ({valid_mask.sum()} pixels)')
        axes[0, 1].axis('off')
        
        # Depth
        depth_vis = np.nan_to_num(depth, nan=0)
        axes[0, 2].imshow(depth_vis, cmap='turbo', vmin=0, vmax=10)
        axes[0, 2].set_title('Depth (meters)')
        axes[0, 2].axis('off')
    
    # Depth histograms per split
    for idx, (split_name, samples) in enumerate(split_data_dict.items()):
        if not samples:
            continue
        
        # Sample some depths for histogram
        all_valid_depths = []
        for sample in samples[:min(10, len(samples))]:
            depth = np.load(dst_dir.parent / sample['depth_path'])
            valid_depths = depth[~np.isnan(depth)]
            all_valid_depths.extend(valid_depths.flatten())
        
        if all_valid_depths:
            axes[1, idx].hist(all_valid_depths, bins=50, alpha=0.7, edgecolor='black')
            axes[1, idx].set_xlabel('Depth (meters)')
            axes[1, idx].set_ylabel('Frequency')
            axes[1, idx].set_title(f'{split_name.capitalize()} Depth Distribution')
            axes[1, idx].grid(alpha=0.3)
    
    plt.tight_layout()
    qc_path = Path('docs/figures/ss4blind_qc.png')
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(qc_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ QC plot saved to {qc_path}")


def main():
    parser = argparse.ArgumentParser(description='Ingest SS4Blind dataset')
    parser.add_argument('--src', type=str, required=True, help='Source directory')
    parser.add_argument('--dst', type=str, required=True, help='Destination directory')
    parser.add_argument('--val_pct', type=float, default=0.10, help='Validation percentage')
    parser.add_argument('--test_pct', type=float, default=0.10, help='Test percentage')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--target_hw', type=int, nargs=2, default=[480, 640], 
                       help='Target height and width')
    args = parser.parse_args()
    
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    target_hw = tuple(args.target_hw)
    
    print("="*60)
    print("SS4Blind Dataset Ingestion")
    print("="*60)
    print(f"Source: {src_dir}")
    print(f"Destination: {dst_dir}")
    print(f"Target size: {target_hw[0]}×{target_hw[1]}")
    print(f"Val: {args.val_pct*100:.0f}%, Test: {args.test_pct*100:.0f}%")
    print("="*60)
    
    # Find files
    print("\n[1/4] Discovering files...")
    files = find_files(src_dir)
    print(f"  RGB: {len(files['rgb'])}")
    print(f"  Depth: {len(files['depth'])}")
    print(f"  Semantic: {len(files['sem'])}")
    
    # Pair samples
    print("\n[2/4] Pairing samples...")
    samples = pair_samples(files)
    print(f"  Valid pairs: {len(samples)}")
    
    if not samples:
        print("❌ No valid RGB-Depth pairs found!")
        return 1
    
    # Split
    print("\n[3/4] Splitting by scene...")
    train, val, test = split_samples(samples, src_dir, args.val_pct, args.test_pct, args.seed)
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")
    
    # Process and save
    print("\n[4/4] Processing and saving...")
    split_data = {}
    
    for split_name, split_list in [('train', train), ('val', val), ('test', test)]:
        rows = []
        print(f"\n  Processing {split_name} split...")
        for idx, sample in enumerate(tqdm(split_list, desc=f"  {split_name}")):
            stem = f"{idx:06d}"
            try:
                row = write_sample(sample, dst_dir, split_name, stem, target_hw)
                rows.append(row)
            except Exception as e:
                print(f"    ⚠ Skipping {sample['rgb'].name}: {e}")
        
        # Write CSV
        csv_path = dst_dir / 'splits' / f'{split_name}.csv'
        write_split_csv(rows, csv_path)
        split_data[split_name] = rows
        print(f"  ✓ {split_name}.csv: {len(rows)} samples")
    
    # Create QC plot
    print("\n[5/5] Creating QC plot...")
    create_qc_plot(dst_dir, split_data)
    
    print("\n" + "="*60)
    print("✓ Ingestion complete!")
    print("="*60)
    print(f"\nDataset structure:")
    print(f"  {dst_dir}/rgb/{{train,val,test}}/")
    print(f"  {dst_dir}/depth/{{train,val,test}}/")
    print(f"  {dst_dir}/sem/{{train,val,test}}/ (if available)")
    print(f"  {dst_dir}/splits/{{train,val,test}}.csv")
    
    # Show example rows
    if split_data['train']:
        print(f"\nExample train.csv rows:")
        for i, row in enumerate(split_data['train'][:3]):
            print(f"  {i+1}. {row}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

