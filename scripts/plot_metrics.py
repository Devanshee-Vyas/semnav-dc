import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

metrics_path = Path("checkpoints/ss4blind_unet/val_metrics.csv")
out_dir = Path("docs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(metrics_path)

# --- RMSE curve ---
plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["rmse"], marker='o', label="Val RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE (m)")
plt.title("Validation RMSE per Epoch")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir/"baseline_rmse.png", dpi=150)
plt.close()

# --- MAE curve ---
plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["mae"], marker='s', color='orange', label="Val MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE (m)")
plt.title("Validation MAE per Epoch")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir/"baseline_mae.png", dpi=150)
plt.close()

print(f"✓ saved plots to {out_dir}")
