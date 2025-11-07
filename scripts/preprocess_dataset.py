"""Ingest SS4Blind(+ARKitScenes) dataset into canonical format."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


# --------------------------
# I/O helpers
# --------------------------
def read_pfm(file_path: Path) -> np.ndarray:
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header not in ('PF', 'Pf'):
            raise ValueError(f'Not a PFM file: {file_path}')
        width, height = map(int, f.readline().decode('utf-8').rstrip().split())
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)
        data = np.fromfile(f, endian + 'f')
    data = data.reshape((height, width, 3)) if header == 'PF' else data.reshape((height, width))
    data = np.flipud(data)
    return data * scale


def read_depth_any(path: Path) -> np.ndarray:
    """Return float32 meters with NaN for invalid."""
    ext = path.suffix.lower()
    if ext == '.npy':
        depth = np.load(path).astype(np.float32)
    elif ext == '.pfm':
        depth = read_pfm(path).astype(np.float32)
    elif ext == '.png':
        # assume uint16 in millimeters
        depth_uint = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if depth_uint is None:
            raise ValueError(f"Could not read depth PNG: {path}")
        if depth_uint.dtype == np.uint16:
            depth = depth_uint.astype(np.float32) / 1000.0
        else:
            # if already float or 8-bit, try best-effort
            depth = depth_uint.astype(np.float32)
    else:
        raise ValueError(f"Unsupported depth format: {ext}")
    depth[depth <= 0] = np.nan
    return depth


# --------------------------
# Scanners
# --------------------------
def find_files(src_dir: Path) -> Dict[str, List[Path]]:
    """Heuristic scan for SS4Blind-like layouts."""
    files = {'rgb': [], 'depth': [], 'sem': []}
    rgb_exts = {'.png', '.jpg', '.jpeg'}
    depth_exts = {'.png', '.npy', '.pfm'}
    sem_exts = {'.png'}

    for f in src_dir.rglob('*'):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        stem = f.stem.lower()
        parent = str(f.parent).lower()

        if ('depth' in stem or 'disp' in stem or 'depth' in parent) and ext in depth_exts:
            files['depth'].append(f)
        elif ('sem' in stem or 'label' in stem or 'semantic' in parent) and ext in sem_exts:
            files['sem'].append(f)
        elif (('color' in stem or 'rgb' in stem) or ('rgb' in parent or 'color' in parent)) and ext in rgb_exts:
            files['rgb'].append(f)
        elif ext in rgb_exts:
            files['rgb'].append(f)
    return files


def scan_arkitscenes(root: Path) -> List[Dict[str, Path]]:
    """
    Pair ARKitScenes raw: raw/{Training,Validation}/{video_id}/vga_wide/*.jpg + lowres_depth/*.png
    """
    rgb_paths = list(root.rglob("*/vga_wide/*.jpg")) + list(root.rglob("*/vga_wide/*.png"))
    depth_paths = list(root.rglob("*/lowres_depth/*.png"))  # can swap to highres_depth if you have it

    def vid_of(p: Path) -> str:
        # .../raw/<Fold>/<video_id>/<subdir>/<file>
        # pick the <video_id> component (third from end)
        parts = p.parts
        return parts[-3]

    rgb_index: Dict[Tuple[str, str], Path] = {}
    for p in rgb_paths:
        rgb_index[(vid_of(p), p.stem.lower())] = p

    samples: List[Dict[str, Path]] = []
    for d in depth_paths:
        key = (vid_of(d), d.stem.lower())
        if key in rgb_index:
            samples.append({"rgb": rgb_index[key], "depth": d})
    return samples


# --------------------------
# Pairing / splitting
# --------------------------
def pair_samples(files: Dict[str, List[Path]]) -> List[Dict[str, Path]]:
    def get_base_id(path: Path) -> str:
        stem = path.stem
        for suffix in ['-color', '-depth', '-rgb', '-disp', '-sem', '-label']:
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
        return stem

    rgb_dict = {get_base_id(f): f for f in files['rgb']}
    depth_dict = {get_base_id(f): f for f in files['depth']}
    sem_dict = {get_base_id(f): f for f in files['sem']}

    samples = []
    for base_id, rgb_path in rgb_dict.items():
        if base_id not in depth_dict:
            continue
        item = {'rgb': rgb_path, 'depth': depth_dict[base_id]}
        if base_id in sem_dict:
            item['sem'] = sem_dict[base_id]
        samples.append(item)
    return samples


def get_scene_id(file_path: Path, src_dir: Path) -> str:
    """
    Group by scene. For ARKitScenes, use <video_id>.
    For generic layouts, use the parent-of-parent if available; else parent.
    """
    rel = file_path.relative_to(src_dir)
    parts = rel.parts
    # try ARKitScenes: .../arkitscenes/raw/<Fold>/<video_id>/...
    if 'arkitscenes' in parts:
        try:
            i = parts.index('arkitscenes')
            # parts[i] = 'arkitscenes', i+1='raw', i+2=<Fold>, i+3=<video_id>
            if len(parts) > i + 3:
                return parts[i + 3]
        except ValueError:
            pass
    # fallback: parent-of-parent if possible
    if len(parts) >= 3:
        return parts[-3]
    if len(parts) >= 2:
        return parts[-2]
    return "default_scene"


def split_samples(samples: List[Dict[str, Path]],
                  src_dir: Path, val_pct: float, test_pct: float, seed: int):
    scene_to_samples: Dict[str, List[Dict[str, Path]]] = {}
    for s in samples:
        scene = get_scene_id(s['rgb'], src_dir)
        scene_to_samples.setdefault(scene, []).append(s)

    scenes = sorted(scene_to_samples.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)

    n_scenes = len(scenes)
    n_test = max(1, int(n_scenes * test_pct))
    n_val = max(1, int(n_scenes * val_pct))
    n_train = max(0, n_scenes - n_test - n_val)

    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train:n_train + n_val]
    test_scenes = scenes[n_train + n_val:]

    train = [s for sc in train_scenes for s in scene_to_samples[sc]]
    val = [s for sc in val_scenes for s in scene_to_samples[sc]]
    test = [s for sc in test_scenes for s in scene_to_samples[sc]]
    return train, val, test


# --------------------------
# Processing
# --------------------------
def resize_safe(img: np.ndarray, target_hw: Tuple[int, int], mode: str = 'bilinear') -> np.ndarray:
    h, w = target_hw
    if mode == 'bilinear':
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    if mode == 'nearest':
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    raise ValueError(f"Unknown mode: {mode}")


def make_sparse_depth(depth_m: np.ndarray, keep_ratio: float, seed: int = 1337):
    """
    Return (sparse_depth with NaN on invalid, valid_mask uint8 in {0,1}).
    """
    valid = ~np.isnan(depth_m)
    mask = np.zeros_like(valid, dtype=np.uint8)
    idx = np.argwhere(valid)
    if idx.size:
        rng = np.random.default_rng(seed)
        k = max(1, int(len(idx) * keep_ratio))
        sel = idx[rng.choice(len(idx), size=k, replace=False)]
        mask[tuple(sel.T)] = 1
    sparse = np.where(mask == 1, depth_m, np.nan).astype(np.float32)
    return sparse, mask


def write_sample(sample: Dict[str, Path], dst_dir: Path, split: str,
                 stem: str, target_hw: Tuple[int, int],
                 sparsity: float = None, save_dense_depth: bool = False) -> Dict[str, str]:
    # RGB
    rgb_bgr = cv2.imread(str(sample['rgb']))
    if rgb_bgr is None:
        raise ValueError(f"Cannot read RGB: {sample['rgb']}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb = resize_safe(rgb, target_hw, 'bilinear')

    # Dense depth (meters)
    depth = read_depth_any(sample['depth'])
    depth = resize_safe(depth, target_hw, 'nearest')

    # Optional semantic
    sem = None
    if 'sem' in sample:
        sem = cv2.imread(str(sample['sem']), cv2.IMREAD_GRAYSCALE)
        if sem is not None:
            sem = resize_safe(sem, target_hw, 'nearest')

    # --- write outputs ---
    # images/
    rgb_out = dst_dir / 'images' / split / f'{stem}.png'
    rgb_out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(rgb_out)

    row = {'rgb_path': str(rgb_out.relative_to(dst_dir))}

    if sparsity is not None:
        sparse, valid_mask = make_sparse_depth(depth, keep_ratio=sparsity)
        ds_out = dst_dir / 'depth_sparse' / split / f'{stem}.npy'
        ds_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(ds_out, sparse)

        vm_out = dst_dir / 'depth_valid_mask' / split / f'{stem}.png'
        vm_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vm_out), (valid_mask * 255).astype(np.uint8))

        row['depth_sparse_path'] = str(ds_out.relative_to(dst_dir))
        row['valid_mask_path'] = str(vm_out.relative_to(dst_dir))

    if save_dense_depth or sparsity is None:
        # always keep dense if user asked, or if sparsity not used
        dd_out = dst_dir / 'depth' / split / f'{stem}.npy'
        dd_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(dd_out, depth.astype(np.float32))
        row['depth_path'] = str(dd_out.relative_to(dst_dir))

    if sem is not None:
        sem_out = dst_dir / 'sem' / split / f'{stem}.png'
        sem_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sem_out), sem)
        row['sem_path'] = str(sem_out.relative_to(dst_dir))

    return row


def write_split_csv(rows: List[Dict[str, str]], csv_path: Path):
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def create_qc_plot(dst_dir: Path, split_data_dict: Dict[str, List[Dict]]):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Example panel from first train sample
    if split_data_dict['train']:
        example = split_data_dict['train'][0]
        rgb = np.array(Image.open(dst_dir / example['rgb_path']))

        # prefer sparse for visualization if present
        depth_path_key = 'depth_sparse_path' if 'depth_sparse_path' in example else 'depth_path'
        depth = np.load(dst_dir / example[depth_path_key])

        axes[0, 0].imshow(rgb); axes[0, 0].set_title('Example RGB'); axes[0, 0].axis('off')
        valid_mask = ~np.isnan(depth)
        axes[0, 1].imshow(valid_mask, cmap='gray'); axes[0, 1].set_title(f'Valid Mask ({valid_mask.sum()} px)'); axes[0, 1].axis('off')
        depth_vis = np.nan_to_num(depth, nan=0)
        axes[0, 2].imshow(depth_vis, cmap='turbo', vmin=0, vmax=10); axes[0, 2].set_title('Depth (m)'); axes[0, 2].axis('off')

    # Histograms
    for idx, (split_name, samples) in enumerate([('train', split_data_dict['train']),
                                                 ('val', split_data_dict['val']),
                                                 ('test', split_data_dict['test'])]):
        if not samples: continue
        all_valid_depths = []
        for sample in samples[:min(10, len(samples))]:
            key = 'depth_sparse_path' if 'depth_sparse_path' in sample else 'depth_path'
            d = np.load(dst_dir / sample[key])
            all_valid_depths.extend(d[~np.isnan(d)].flatten())
        if len(all_valid_depths):
            axes[1, idx].hist(all_valid_depths, bins=50, alpha=0.7, edgecolor='black')
            axes[1, idx].set_xlabel('Depth (m)'); axes[1, idx].set_ylabel('Freq')
            axes[1, idx].set_title(f'{split_name.capitalize()} Depth Distribution')
            axes[1, idx].grid(alpha=0.3)

    plt.tight_layout()
    qc_path = Path('docs/figures/ss4blind_qc.png')
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(qc_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n✓ QC plot saved to {qc_path}")


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description='Ingest SS4Blind(+ARKitScenes) dataset')
    parser.add_argument('--src', type=str, required=True, help='Source directory root (contains ss4blind/arkitscenes etc.)')
    parser.add_argument('--dst', type=str, required=True, help='Destination directory')
    parser.add_argument('--val_pct', type=float, default=0.10)
    parser.add_argument('--test_pct', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--target_hw', type=int, nargs=2, default=[480, 640], help='H W')
    parser.add_argument('--sparsity', type=float, default=None, help='Keep ratio for sparse depth (e.g., 0.03). If None, no sparsification.')
    parser.add_argument('--save_dense_depth', action='store_true', help='Also save dense depth .npy')
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    target_hw = tuple(args.target_hw)

    print("="*60)
    print("Dataset Ingestion (SS4Blind + ARKitScenes)")
    print("="*60)
    print(f"Source: {src_dir}")
    print(f"Destination: {dst_dir}")
    print(f"Target size: {target_hw[0]}×{target_hw[1]}")
    print(f"Val: {args.val_pct*100:.0f}%, Test: {args.test_pct*100:.0f}%")
    print(f"Sparsity: {args.sparsity}, Save dense: {args.save_dense_depth}")
    print("="*60)

    print("\n[1/4] Discovering samples...")
    samples: List[Dict[str, Path]] = []

    # ARKitScenes under: <src>/arkitscenes/raw/...
    ark_root = src_dir / "arkitscenes" / "raw"
    if ark_root.exists():
        ark_samples = scan_arkitscenes(ark_root)
        print(f"  ARKitScenes pairs: {len(ark_samples)}")
        samples += ark_samples
    else:
        print("  ARKitScenes not found (skip)")

    # SS4Blind-style generic scan
    files = find_files(src_dir)
    print(f"  SS4Blind scan — RGB:{len(files['rgb'])} Depth:{len(files['depth'])} Sem:{len(files['sem'])}")
    ss_pairs = pair_samples(files)
    print(f"  SS4Blind pairs: {len(ss_pairs)}")
    samples += ss_pairs

    total_pairs = len(samples)
    print(f"\n[2/4] Pairing complete, total valid pairs: {total_pairs}")
    if total_pairs == 0:
        print("❌ No valid RGB-Depth pairs found!")
        return 1

    print("\n[3/4] Splitting by scene...")
    train, val, test = split_samples(samples, src_dir, args.val_pct, args.test_pct, args.seed)
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    print("\n[4/4] Processing & saving...")
    split_data = {}
    for split_name, split_list in [('train', train), ('val', val), ('test', test)]:
        rows = []
        print(f"\n  Processing {split_name}...")
        for idx, s in enumerate(tqdm(split_list, desc=f"  {split_name}")):
            stem = f"{idx:06d}"
            try:
                row = write_sample(s, dst_dir, split_name, stem, target_hw,
                                   sparsity=args.sparsity, save_dense_depth=args.save_dense_depth)
                rows.append(row)
            except Exception as e:
                print(f"    ⚠ Skipping {s['rgb'].name}: {e}")
        csv_path = dst_dir / 'splits' / f'{split_name}.csv'
        write_split_csv(rows, csv_path)
        split_data[split_name] = rows
        print(f"  ✓ {split_name}.csv: {len(rows)} samples")

    print("\n[5/5] Creating QC plot...")
    create_qc_plot(dst_dir, split_data)

    print("\n" + "="*60)
    print("✓ Ingestion complete!")
    print("="*60)
    print(f"Output structure:")
    print(f"  {dst_dir}/images/{{train,val,test}}/")
    print(f"  {dst_dir}/depth_sparse/{{train,val,test}}/ (if --sparsity)")
    print(f"  {dst_dir}/depth_valid_mask/{{train,val,test}}/ (if --sparsity)")
    print(f"  {dst_dir}/depth/{{train,val,test}}/ (if --save_dense_depth or sparsity None)")
    print(f"  {dst_dir}/sem/{{train,val,test}}/ (if available)")
    print(f"  {dst_dir}/splits/{{train,val,test}}.csv")
    if split_data['train'][:3]:
        print("\nExample train.csv rows:")
        for i, row in enumerate(split_data['train'][:3]):
            print(f"  {i+1}. {row}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
