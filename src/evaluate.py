"""Evaluation script for depth completion models."""
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data import make_dataset
from models.unet_baseline import UNetBaseline
from models.semnav_dc import SemNavDC
from metrics.depth import batch_compute_metrics
from metrics.navigation import batch_compute_navigation_metrics
from utils.io_utils import load_config, load_checkpoint
from utils.viz import save_depth_panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Paths
    exp_name = cfg['exp_name']
    output_dir = Path('results') / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset using factory
    dataset_name = cfg.get('dataset_name', 'nyu')  # default to nyu for backward compatibility
    dataset = make_dataset(dataset_name, args.split, cfg)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # Model
    if cfg['model'] == 'unet_baseline':
        model = UNetBaseline()
    elif cfg['model'] == 'semnav_dc':
        model = SemNavDC(predict_freespace=True)
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")
    
    # Load checkpoint
    load_checkpoint(args.ckpt, model)
    model.eval()
    
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Evaluating on {args.split} split ({len(dataset)} samples)")
    
    # Evaluate
    depth_metrics_list = []
    nav_metrics_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
            rgb = batch['rgb']
            # Handle both NYU and SS4Blind formats
            sparse = batch.get('sparse', batch.get('sparse_depth'))
            mask = batch.get('mask', batch.get('sparse_mask'))
            gt = batch.get('gt', batch.get('depth'))
            
            # Forward
            x = torch.cat([rgb, sparse, mask], dim=1)
            
            if cfg['model'] == 'unet_baseline':
                pred = model(x)
            else:
                output = model(x)
                pred = output['depth']
            
            # Depth metrics
            d_metrics = batch_compute_metrics(pred, gt, mask)
            depth_metrics_list.append(d_metrics)
            
            # Navigation metrics
            nav_metrics = batch_compute_navigation_metrics(pred, gt)
            nav_metrics_list.append(nav_metrics)
            
            # Save qualitative examples (first 16)
            if i < 16:
                save_depth_panel(
                    rgb, sparse, pred, gt,
                    str(output_dir / f'sample_{i:03d}.png')
                )
    
    # Average metrics
    avg_depth_metrics = {
        k: np.mean([m[k] for m in depth_metrics_list])
        for k in depth_metrics_list[0].keys()
    }
    
    avg_nav_metrics = {
        k: np.mean([m[k] for m in nav_metrics_list])
        for k in nav_metrics_list[0].keys()
    }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print("\nDepth Metrics:")
    print(f"  RMSE:  {avg_depth_metrics['rmse']:.4f} m")
    print(f"  MAE:   {avg_depth_metrics['mae']:.4f} m")
    print(f"  iRMSE: {avg_depth_metrics['irmse']:.4f}")
    print(f"  iMAE:  {avg_depth_metrics['imae']:.4f}")
    
    print("\nNavigation Metrics:")
    print(f"  Free Space IoU:      {avg_nav_metrics['freespace_iou']:.4f}")
    print(f"  Obstacle Recall@2m:  {avg_nav_metrics['obstacle_recall_2m']:.4f}")
    print("="*60)
    
    # Save results to markdown
    results_md = output_dir / 'results.md'
    with open(results_md, 'w') as f:
        f.write(f"# Evaluation Results: {exp_name}\n\n")
        f.write(f"**Split:** {args.split}\n\n")
        f.write(f"**Checkpoint:** {args.ckpt}\n\n")
        
        f.write("## Depth Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| RMSE   | {avg_depth_metrics['rmse']:.4f} m |\n")
        f.write(f"| MAE    | {avg_depth_metrics['mae']:.4f} m |\n")
        f.write(f"| iRMSE  | {avg_depth_metrics['irmse']:.4f} |\n")
        f.write(f"| iMAE   | {avg_depth_metrics['imae']:.4f} |\n")
        
        f.write("\n## Navigation Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Free Space IoU     | {avg_nav_metrics['freespace_iou']:.4f} |\n")
        f.write(f"| Obstacle Recall@2m | {avg_nav_metrics['obstacle_recall_2m']:.4f} |\n")
    
    print(f"\nResults saved to: {results_md}")
    print(f"Qualitative samples saved to: {output_dir}/")


if __name__ == '__main__':
    main()
