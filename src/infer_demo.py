"""Demo inference script for depth completion."""
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.unet_baseline import UNetBaseline
from models.semnav_dc import SemNavDC
from data.transforms import normalize_rgb
from utils.io_utils import load_config, load_checkpoint
from utils.viz import colorize_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--input', type=str, required=True, help='Image or folder')
    parser.add_argument('--output', type=str, default='output')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
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
    
    print(f"Loaded model from: {args.ckpt}")
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_dir():
        image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    else:
        image_files = [input_path]
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(image_files)} images...")
    
    # Process
    with torch.no_grad():
        for img_path in tqdm(image_files):
            # Load and preprocess
            rgb = np.array(Image.open(img_path).convert('RGB'))
            H, W = rgb.shape[:2]
            
            # Resize to model input size
            rgb_resized = np.array(Image.fromarray(rgb).resize((320, 240)))
            
            # Normalize
            rgb_norm = normalize_rgb(rgb_resized)
            rgb_tensor = torch.from_numpy(rgb_norm).unsqueeze(0).float()
            
            # Create dummy sparse and mask (all zeros for RGB-only inference)
            sparse = torch.zeros(1, 1, 240, 320)
            mask = torch.zeros(1, 1, 240, 320)
            
            # Concatenate
            x = torch.cat([rgb_tensor, sparse, mask], dim=1)
            
            # Predict
            if cfg['model'] == 'unet_baseline':
                pred = model(x)
            else:
                output = model(x)
                pred = output['depth']
            
            # Post-process
            pred_np = pred[0, 0].cpu().numpy()
            
            # Colorize
            depth_colored = colorize_depth(pred_np, vmax=10)
            
            # Generate navigation cues
            cues = generate_navigation_cues(pred_np)
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].imshow(rgb_resized)
            axes[0].set_title('Input RGB')
            axes[0].axis('off')
            
            axes[1].imshow(depth_colored)
            axes[1].set_title(f'Predicted Depth\n{cues}')
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # Save
            output_path = output_dir / f"{img_path.stem}_depth.png"
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"  {img_path.name}: {cues}")
    
    print(f"\nOutputs saved to: {output_dir}/")


def generate_navigation_cues(depth: np.ndarray) -> str:
    """
    Generate simple navigation cues from depth map.
    
    Args:
        depth: (H, W) depth array
        
    Returns:
        String with navigation guidance
    """
    H, W = depth.shape
    
    # Center region (for forward path)
    center_region = depth[H//2:, W//3:2*W//3]
    
    if center_region.size == 0:
        return "No depth data"
    
    median_depth = np.median(center_region)
    min_depth = np.min(center_region)
    
    if min_depth < 1.0:
        return f"⚠ Obstacle at {min_depth:.1f}m"
    elif median_depth > 3.0:
        return f"✓ Clear path {median_depth:.1f}m"
    else:
        return f"→ Navigate carefully {median_depth:.1f}m"


if __name__ == '__main__':
    main()
