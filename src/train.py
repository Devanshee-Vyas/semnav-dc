"""Training script for depth completion models."""
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import make_dataset
from models.unet_baseline import UNetBaseline
from models.semnav_dc import SemNavDC
from losses.depth import DepthLoss
from losses.boundary import BoundaryIoULoss
from metrics.depth import batch_compute_metrics
from metrics.navigation import batch_compute_navigation_metrics
from utils.io_utils import load_config, save_checkpoint, write_csv_log
from utils.viz import save_depth_panel
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    set_seed(42)
    
    # Paths
    exp_name = cfg['exp_name']
    ckpt_dir = Path('checkpoints') / exp_name
    log_dir = Path('logs') / exp_name
    viz_dir = Path('docs/figures') / exp_name
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset using factory
    dataset_name = cfg.get('dataset_name', 'nyu')  # default to nyu for backward compatibility
    train_dataset = make_dataset(dataset_name, 'train', cfg)
    val_dataset = make_dataset(dataset_name, 'val', cfg)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=False
    )
    
    # Model
    if cfg['model'] == 'unet_baseline':
        model = UNetBaseline()
    elif cfg['model'] == 'semnav_dc':
        model = SemNavDC(predict_freespace=True)
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")
    
    # Loss
    loss_cfg = cfg['loss']
    depth_loss = DepthLoss(
        l1_weight=loss_cfg.get('l1', 1.0),
        grad_weight=loss_cfg.get('grad', 0.5),
        ssim_weight=loss_cfg.get('ssim', 0.0)
    )
    boundary_loss = BoundaryIoULoss() if loss_cfg.get('boundary_iou', 0) > 0 else None
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg.get('weight_decay', 0.0001)
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs']
    )
    
    # Training loop
    best_rmse = float('inf')
    
    for epoch in range(1, cfg['epochs'] + 1):
        print(f"\nEpoch {epoch}/{cfg['epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, depth_loss, boundary_loss, optimizer, cfg, epoch
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, depth_loss, cfg, epoch, viz_dir)
        
        # Scheduler step
        scheduler.step()
        
        # Log
        log_data = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae']
        }
        write_csv_log(log_dir / 'metrics.csv', log_data)
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val RMSE: {val_metrics['rmse']:.4f}")
        
        # Save best checkpoint
        if val_metrics['rmse'] < best_rmse:
            best_rmse = val_metrics['rmse']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(ckpt_dir / 'best_rmse.ckpt')
            )
            print(f"✓ Saved best checkpoint (RMSE: {best_rmse:.4f})")
    
    print(f"\nTraining completed! Best RMSE: {best_rmse:.4f}")


def train_epoch(model, loader, depth_loss, boundary_loss, optimizer, cfg, epoch):
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for batch in pbar:
        rgb = batch['rgb']
        # Handle both NYU (sparse/mask/gt) and SS4Blind (sparse_depth/sparse_mask/depth) formats
        sparse = batch.get('sparse', batch.get('sparse_depth'))
        mask = batch.get('mask', batch.get('sparse_mask'))
        gt = batch.get('gt', batch.get('depth'))
        
        # Forward
        x = torch.cat([rgb, sparse, mask], dim=1)  # (B, 5, H, W)
        
        if cfg['model'] == 'unet_baseline':
            pred = model(x)
            output = {'depth': pred}
        else:
            output = model(x)
            pred = output['depth']
        
        # Compute loss
        loss_dict = depth_loss(pred, gt, mask)
        loss = loss_dict['total']
        
        # Add boundary loss if enabled
        if boundary_loss is not None:
            b_loss = boundary_loss(pred, gt, mask)
            loss += cfg['loss'].get('boundary_iou', 0.2) * b_loss
            loss_dict['boundary'] = b_loss
        
        # Add freespace loss if available
        if 'freespace' in output and cfg['loss'].get('freespace_bce', 0) > 0:
            # Simple BCE with pseudo GT (depth > 1.5m in lower half)
            B, _, H, W = gt.shape
            pseudo_freespace = torch.zeros_like(gt)
            pseudo_freespace[:, :, H//2:, :] = (gt[:, :, H//2:, :] > 1.5).float()
            
            fs_loss = torch.nn.functional.binary_cross_entropy(
                output['freespace'], pseudo_freespace
            )
            loss += cfg['loss']['freespace_bce'] * fs_loss
            loss_dict['freespace'] = fs_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return {'loss': total_loss / num_batches}


def validate_epoch(model, loader, depth_loss, cfg, epoch, viz_dir):
    model.eval()
    
    total_loss = 0.0
    all_metrics = []
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Validation")):
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
            
            # Loss
            loss_dict = depth_loss(pred, gt, mask)
            loss = loss_dict['total']
            
            total_loss += loss.item()
            num_batches += 1
            
            # Metrics
            metrics = batch_compute_metrics(pred, gt, mask)
            all_metrics.append(metrics)
            
            # Save visualization
            if i == 0 and epoch % cfg.get('viz_every', 2) == 0:
                save_depth_panel(
                    rgb, sparse, pred, gt,
                    str(viz_dir / f'epoch_{epoch:03d}.png')
                )
    
    # Average metrics
    avg_metrics = {
        k: sum(m[k] for m in all_metrics) / len(all_metrics)
        for k in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / num_batches
    
    return avg_metrics


if __name__ == '__main__':
    main()
