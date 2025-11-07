# 🧭 SemNav-DC Baseline — UNet Depth Completion for Assistive Navigation

Baseline implementation of **semantic-guided sparse depth completion** for assistive navigation systems.  
This repository reproduces the **UNet baseline** on the combined **SS4Blind + ARKitScenes** dataset, generating dense depth maps from RGB and sparse depth inputs.

---

## 🚀 Overview

This project provides a complete end-to-end pipeline for:

- Processing and merging multiple RGB-D datasets  
- Generating sparse depth and validity masks  
- Training a baseline UNet depth completion model  
- Evaluating RMSE/MAE metrics and visualizing predictions  

---

## 📦 Environment Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/semnav-dc.git
cd semnav-dc
```

### 2️⃣ Create a Python environment
```bash
conda create -n semnav python=3.11 -y
conda activate semnav
pip install -r requirements.txt
```

💡 On Apple Silicon (M1/M2/M3), PyTorch automatically uses MPS (Metal GPU) acceleration.

---

## 🗂 Dataset Preparation

### 🧩 SS4Blind Dataset

Download from the official repository:  
https://github.com/elnino9ykl/SS4Blind

Required sub-datasets (must contain RGB + Depth):
- rgbdss/
- terrain/

Optional semantic datasets:
- crosswalk/
- curb/
- gardens_point/

Place them in:
```
data/ss4blind/raw/
├── rgbdss/
├── terrain/
├── crosswalk/
├── curb/
└── gardens_point/
```

---

### 🍎 ARKitScenes Dataset (Apple)

1. Visit: https://machinelearning.apple.com/research/arkitscenes  
   Agree to the license terms.  
2. Download several `video_id` folders (e.g., `40753679`, `40753686`, `40776203`) from the Raw Dataset section.  
3. Extract them into:
```
data/ss4blind/raw/arkitscenes/raw/Training/<video_id>/{vga_wide,lowres_depth,...}
```

⚠️ For a quick demo, downloading only one scene (~1 GB) is sufficient.

---

### 📦 Processed Dataset (Quick Start)

To skip preprocessing, you can directly download the preprocessed dataset (~6.8 GB):

[📥 Download Processed Dataset (Google Drive)](https://drive.google.com/file/d/XXXXXXXXXXXX/view?usp=sharing)

After downloading, unzip it into:
src/data/ss4blind/processed/

Then you can start training directly:
```bash
python -m src.train_unet \
  --data_root src/data/ss4blind/processed \
  --epochs 10 \
  --batch_size 8 \
  --lr 3e-4
```

---

## ⚙️ Data Preprocessing

Combine SS4Blind and ARKitScenes into a unified processed format:
```bash
python -m scripts.preprocess_dataset   
    --src data/ss4blind/raw   
    --dst src/data/ss4blind/processed   
    --target_hw 480 640   
    --val_pct 0.10 
    --test_pct 0.10   
    --sparsity 0.03 
    --save_dense_depth
```

Output structure:
```
src/data/ss4blind/processed/
├── images/{train,val,test}/
├── depth_sparse/{train,val,test}/
├── depth_valid_mask/{train,val,test}/
├── depth/{train,val,test}/
└── splits/train.csv, val.csv, test.csv
```

---

## 🧠 Training the Baseline (UNet)

Train the UNet baseline model using the processed dataset:
```bash
python -m src.train_unet   
    --data_root src/data/ss4blind/processed   
    --epochs 10   
    --batch_size 8   
    --lr 3e-4   
    --outdir checkpoints/ss4blind_unet  
    --subset_train 0 
    --subset_val 0
```

- Automatically uses MPS or CPU  
- Logs and best checkpoint are saved in `checkpoints/ss4blind_unet/`

---

## 📈 Generating Results

### 1️⃣ Plot Validation Curves
```bash
python -m scripts.plot_metrics
```
Outputs:
- docs/figures/baseline_rmse.png  
- docs/figures/baseline_mae.png

### 2️⃣ Export Visualization Panels
```bash
python -m scripts.export_panels
```
Creates qualitative examples in:
```
docs/figures/baseline_examples/panel_*.png
```
Each panel shows: RGB | Sparse | Predicted | Ground Truth

---

## 🧪 Evaluate on Test Set
```bash
python -m scripts.eval_on_test
```
Example output:
```
TEST  RMSE=0.1327  MAE=0.0914
```

---

## 📊 Baseline Metrics Summary

| Split | RMSE (m) | MAE (m) |
|:------|:---------:|:-------:|
| Validation | 0.1245 | 0.0830 |
| Test | 0.1327 | 0.0914 |

Results are stored in:  
docs/tables/baseline_metrics.csv

---

## 🧩 Quick Start Recap
```bash
# 1. Download datasets → data/ss4blind/raw/
# 2. Preprocess
python -m scripts.preprocess_dataset 
    --src data/ss4blind/raw 
    --dst src/data/ss4blind/processed 
    --target_hw 480 640 
    --sparsity 0.03 
    --save_dense_depth
# 3. Train baseline
python -m src.train_unet 
    --data_root src/data/ss4blind/processed 
    --epochs 10 
    --batch_size 8 
    --lr 3e-4 
    --outdir checkpoints/ss4blind_unet
# 4. Visualize results
python -m scripts.plot_metrics
python -m scripts.export_panels
python -m scripts.eval_on_test
```

---

## 🚫 Git Ignore for Large Files

Datasets and model checkpoints are not included in this repository.  
Make sure `.gitignore` contains:
```
data/
checkpoints/
outputs/
*.pth
*.npy
```

---

## 📁 Project Structure
```
semnav-dc/
├── src/
│   ├── data/ss4blind_dataset.py
│   ├── models/unet_baseline.py
│   └── train_unet.py
├── scripts/
│   ├── preprocess_dataset.py
│   ├── plot_metrics.py
│   ├── export_panels.py
│   └── eval_on_test.py
├── checkpoints/
├── docs/
│   ├── figures/
│   └── tables/
├── requirements.txt
└── README.md
```

---

## 🏁 Final Deliverables

After completing the steps above, you will have:

✅ Trained baseline model → `checkpoints/ss4blind_unet/best.pth`  
📈 Validation curves → `docs/figures/baseline_rmse.png`, `baseline_mae.png`  
🖼 Visualization panels → `docs/figures/baseline_examples/panel_*.png`  
📊 Metrics table → `docs/tables/baseline_metrics.csv`

This completes the UNet baseline reproduction for the SemNav-DC project.

---

✅ **How to use:**  
Copy this entire text and save it as `README.md` in your project root.  
Once pushed to GitHub, it will render beautifully with all emojis, headings, and code blocks.
