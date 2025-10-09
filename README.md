# SemNav-DC: Semantic Depth Completion for Blind Navigation

Deep Learning course project implementing semantic-guided sparse depth completion for assistive navigation systems.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project addresses depth completion for visually impaired navigation by combining semantic segmentation with sparse depth measurements. We demonstrate that semantic object understanding improves depth edge accuracy, which is critical for obstacle detection in assistive navigation systems.

**Key Features:**
- Semantic-guided depth completion architecture (SemNav-DC)
- Navigation-specific evaluation metrics (Free Space IoU, Obstacle Recall)
- Boundary-aware loss functions for sharp edge preservation
- Trained and evaluated on real-world SS4Blind navigation dataset

---

## Architecture

### Baseline: UNet with Multi-Scale Features

- **Encoder:** ResNet-34 (pretrained on ImageNet)
- **Input:** 5 channels (RGB + sparse depth + validity mask)
- **Decoder:** 4-level upsampling with skip connections
- **Output:** Dense depth map

### Advanced: SemNav-DC

- **Semantic Branch:** Frozen DeepLabV3 for semantic features
- **Geometric Branch:** UNet baseline for depth processing
- **Fusion:** Learned gating mechanisms for semantic-geometric integration
- **Losses:** L1, Gradient, SSIM, Boundary IoU, Freespace BCE

---

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Setup

```bash
# Clone repository
git clone https://github.com/Devanshee-Vyas/semnav-dc.git
cd semnav-dc

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU version
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Dataset Preparation

### SS4Blind Dataset

Place your raw SS4Blind dataset in `data/raw/ss4blind/` and run:

```bash
python scripts/preprocess_dataset.py \
    --src data/raw/ss4blind \
    --dst data/ss4blind \
    --val_pct 0.10 \
    --test_pct 0.10 \
    --seed 1337 \
    --target_hw 480 640
```

This will:
- Process RGB and depth images
- Handle multiple depth formats (PNG, NPY, PFM)
- Create train/val/test splits (scene-based to prevent leakage)
- Generate CSV index files

### Validation

Verify dataset paths are correct:

```bash
python scripts/validate_dataset.py \
    --splits_dir data/ss4blind/splits \
    --root data/ss4blind
```

---

## Training

### UNet Baseline

```bash
python src/train.py --config configs/unet_baseline.yaml
```

### SemNav-DC

```bash
python src/train.py --config configs/semnav_dc.yaml
```

**Training Configuration:**
- Batch size: 2 (CPU), 8-16 (GPU)
- Optimizer: Adam (lr=1e-3, weight decay=1e-4)
- Epochs: 5-30
- Mixed precision: Supported via AMP

---

## Evaluation

```bash
python src/evaluate.py \
    --config configs/unet_baseline.yaml \
    --ckpt checkpoints/ss4blind_unet/best_rmse.ckpt
```

**Metrics Computed:**
- Standard: RMSE, MAE, iRMSE, iMAE
- Navigation-Specific: Free Space IoU, Obstacle Recall@2m

---

## Results

### SS4Blind Dataset (Real-World Navigation)

| Model | RMSE (m) | MAE (m) | Free Space IoU | Obstacle Recall@2m |
|-------|----------|---------|----------------|---------------------|
| UNet Baseline | 0.848 | 0.409 | 0.979 | 0.003 |
| SemNav-DC | TBD | TBD | TBD | TBD |

*Training: 184 samples, Validation: 100 samples, Test: 100 samples*

### Qualitative Results

Sample predictions showing RGB input, sparse depth, prediction, and ground truth are saved in `results/` directory after evaluation.

---

## Project Structure

```
semnav-dc/
├── configs/              # Experiment configurations
│   ├── unet_baseline.yaml
│   └── semnav_dc.yaml
├── data/                 # Datasets (gitignored)
├── src/
│   ├── data/            # Dataset loaders
│   │   ├── ss4blind.py
│   │   └── transforms.py
│   ├── models/          # Architecture definitions
│   │   ├── unet_baseline.py
│   │   └── semnav_dc.py
│   ├── losses/          # Loss functions
│   │   ├── depth.py
│   │   └── boundary.py
│   ├── metrics/         # Evaluation metrics
│   │   ├── depth.py
│   │   └── navigation.py
│   ├── utils/           # Utilities
│   ├── train.py         # Training script
│   └── evaluate.py      # Evaluation script
├── scripts/
│   ├── preprocess_dataset.py    # Dataset ingestion
│   └── validate_dataset.py      # Path validation
├── requirements.txt     # Python dependencies
└── PROJECT_DOCUMENTATION.md  # Detailed technical documentation
```

---

## Technical Documentation

For comprehensive technical details including:
- Problem formulation and motivation
- Architecture design and innovations
- Loss function derivations
- Experimental setup and analysis

Please see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{vyas2025semnav,
  title={SemNav-DC: Semantic Depth Completion for Blind Navigation},
  author={Vyas, Devanshee},
  year={2025},
  note={Deep Learning Course Project}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- **Dataset:** SS4Blind RGB-D-SS for assistive navigation research
- **Architectures:** UNet (Ronneberger et al., MICCAI 2015), ResNet (He et al., CVPR 2016), DeepLabV3 (Chen et al., arXiv 2017)
- **Libraries:** PyTorch, timm, OpenCV

---

## Contact

**Author:** Devanshee Vyas  
**Course:** Deep Learning, Semester 3  
**GitHub:** [@Devanshee-Vyas](https://github.com/Devanshee-Vyas)

For questions or collaboration opportunities, please open an issue on GitHub.
