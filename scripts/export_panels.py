import torch, numpy as np
from PIL import Image
from pathlib import Path
from src.data.ss4blind_dataset import SS4BlindDataset
from src.models.unet_baseline import UNetSmall

root = Path("src/data/ss4blind/processed")
ckpt = Path("checkpoints/ss4blind_unet/best.pth")

out_dir = Path("docs/figures/baseline_examples")
out_dir.mkdir(parents=True, exist_ok=True)

# Dataset & model
ds = SS4BlindDataset(root, "val", use_sparse=True)
model = UNetSmall(in_ch=5, out_ch=1)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

def colorize_depth(d, vmax=10.0):
    d = np.nan_to_num(d, nan=0.0)
    x = np.clip(d / vmax, 0, 1)
    x = (x * 255).astype(np.uint8)
    return np.stack([x, x, x], axis=-1)  

for idx in range(min(10, len(ds))):
    b = ds[idx]
    x = b["x"].unsqueeze(0)
    with torch.no_grad():
        pred = model(x).squeeze().numpy()
    rgb = (b["x"][:3].permute(1,2,0).numpy() * 255).astype(np.uint8)
    sparse = b["x"][3].numpy()
    gt = b["depth_gt"].numpy()

    imgs = [
        Image.fromarray(rgb),
        Image.fromarray(colorize_depth(sparse)),
        Image.fromarray(colorize_depth(pred)),
        Image.fromarray(colorize_depth(gt))
    ]
    titles = ["RGB", "Sparse Depth", "Predicted Depth", "Ground Truth"]

    w, h = imgs[0].size
    panel = Image.new("RGB", (w * 4, h + 30), (255,255,255))
    for i, im in enumerate(imgs):
        panel.paste(im, (i * w, 0))
    panel.save(out_dir / f"panel_{idx:02d}.png")

print(f"✓ saved {len(list(out_dir.glob('*.png')))} panels to {out_dir}")
