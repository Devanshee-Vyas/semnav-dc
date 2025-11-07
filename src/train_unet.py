import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from src.data.ss4blind_dataset import SS4BlindDataset
from src.models.unet_baseline import UNetSmall
import os, argparse, numpy as np, pandas as pd
import time

# ---------- masked metrics（會自動對齊維度 + 正規化 valid） ----------
def _align(pred, gt, valid):
    # to [B,1,H,W]
    if pred.ndim == 3:  pred = pred.unsqueeze(1)
    if gt.ndim == 3:    gt = gt.unsqueeze(1)
    if valid.ndim == 3: valid = valid.unsqueeze(1)
    # valid 轉 0/1
    if valid.max() > 1.5:
        valid = (valid > 0).float()
    return pred, gt, valid

def masked_rmse(pred, gt, valid):
    pred, gt, valid = _align(pred, gt, valid)
    denom = valid.sum().clamp_min(1.0)
    return torch.sqrt(((pred - gt) ** 2 * valid).sum() / denom)

def masked_mae(pred, gt, valid):
    pred, gt, valid = _align(pred, gt, valid)
    denom = valid.sum().clamp_min(1.0)
    return ((pred - gt).abs() * valid).sum() / denom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="src/data/ss4blind/processed")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)  # 小一點較穩
    ap.add_argument("--outdir", default="checkpoints/ss4blind_unet")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # dataset / dataloader
    train_ds = SS4BlindDataset(args.data_root, "train", use_sparse=True)
    val_ds   = SS4BlindDataset(args.data_root, "val",   use_sparse=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0, pin_memory=False)

    # device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device, "| mps available:", torch.backends.mps.is_available())

    # model / optim / loss
    model = UNetSmall(in_ch=5, out_ch=1).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.SmoothL1Loss(reduction="none", beta=0.1)  # Huber

    best_rmse = float("inf")
    rows = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()

        for i, b in enumerate(train_dl, 1):
            x = b["x"].to(device)
            gt = b["depth_gt"].to(device)      # [B,H,W]
            valid = b["valid_gt"].to(device)   # [B,H,W] 或 0/255

            pred = model(x)                    # [B,1,H,W]
            P, G, V = _align(pred, gt, valid)

            loss = (crit(P, G) * V).sum() / (V.sum() + 1e-8)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 200 == 0:
                elapsed = time.time() - start
                print(f"  epoch {epoch} | iter {i}/{len(train_dl)} | {elapsed/i:.3f}s/iter | loss {loss.item():.4f}")

        # ------------ validate ------------
        model.eval()
        rmse_list, mae_list = [], []
        with torch.no_grad():
            for b in val_dl:
                x = b["x"].to(device)
                gt = b["depth_gt"].to(device)
                valid = b["valid_gt"].to(device)
                pred = model(x)
                rmse_list.append(masked_rmse(pred, gt, valid).item())
                mae_list.append(masked_mae(pred, gt, valid).item())

        rmse_v = float(np.mean(rmse_list))
        mae_v  = float(np.mean(mae_list))
        rows.append({"epoch": epoch, "rmse": rmse_v, "mae": mae_v})

        end = time.time()
        print(f"[{epoch}] epoch time: {end - start:.1f}s  | val RMSE={rmse_v:.4f}  MAE={mae_v:.4f}")

        # save best
        if rmse_v < best_rmse:
            best_rmse = rmse_v
            torch.save(model.state_dict(), os.path.join(args.outdir, "best.pth"))

    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "val_metrics.csv"), index=False)
    print("✓ training done; best.ckpt saved")

if __name__ == "__main__":
    main()
