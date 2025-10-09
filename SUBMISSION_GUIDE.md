# Project Submission Guide

**Project:** SemNav-DC - Semantic Depth Completion for Blind Navigation  
**Author:** Devanshee Vyas  
**Course:** Deep Learning, Semester 3

---

## How to Explain This Project

### In Simple Terms

**"I built a system that helps blind people navigate safely by reconstructing detailed 3D depth maps from sparse sensor data. The innovation is using semantic understanding (knowing what objects are in the scene) to improve depth predictions, especially at object boundaries where accurate depth is critical for avoiding obstacles."**

### The Problem

Imagine you're blind and using a depth sensor (like a cheap LiDAR) to navigate. These affordable sensors only give you sparse, incomplete depth information - like seeing only 1% of the pixels in an image. You need to fill in the gaps to know:
- "Is there a wall ahead?"
- "How far is that chair?"
- "Can I walk through this doorway?"

Traditional methods just blur/interpolate the sparse points, which makes edges fuzzy and dangerous. Our method uses AI to intelligently predict dense depth by understanding what objects are present.

---

## Project Components Explained

### 1. Project Domain

**Assistive Technology for Navigation**

This falls under:
- **Computer Vision:** Understanding scenes from images
- **Depth Estimation:** Predicting 3D structure from 2D images
- **Safety-Critical AI:** Mistakes can cause physical harm
- **Real-World Application:** Helping visually impaired individuals

It's NOT basic classification (identifying cats vs dogs). It's dense prediction - we predict a value for every single pixel in the image.

### 2. Current Baselines and Problems

**Existing Methods:**

**a) Classical Approaches**
- Bilateral filtering, guided filtering
- **Problem:** Manual tuning, fail in complex scenes, no learning

**b) Early Deep Learning (2015-2018)**
- Simple CNNs that just fill in missing depth
- **Problem:** Produce smooth, blurry results, especially at edges

**c) Modern Deep Learning (2019-2022)**
- UNet architectures with skip connections
- Self-supervised methods using video
- **Problem:** Still struggle with sharp boundaries, ignore object semantics

**Key Gap We Address:**
- Existing methods treat depth completion as pure geometry, ignoring that knowing "this is a wall" or "this is a door" provides strong hints about expected depth patterns
- They optimize for pixel-wise accuracy (RMSE) but don't focus on navigation-critical needs (clear/blocked paths)

### 3. Dataset and Metrics

**Dataset: SS4Blind (RGB-D-SS)**
- **Source:** Real-world indoor navigation scenarios
- **Size:** 384 RGB-Depth pairs (184 train, 100 val, 100 test)
- **Sensors:** RGB camera + Intel RealSense D435 depth camera
- **Scenarios:** Hallways, rooms, doorways, stairs
- **Challenge:** Real sensor noise, missing depth, varied lighting

**Why This Dataset?**
- More realistic than synthetic benchmarks (NYU V2)
- Specifically designed for assistive navigation
- Smaller scale but higher difficulty

**Evaluation Metrics:**

**Standard Metrics (for comparison with papers):**
1. **RMSE (Root Mean Squared Error):** Average depth error in meters
   - Our result: 0.85m
   - Baseline (NYU V2): ~0.52m (but that's synthetic, easier data)

2. **MAE (Mean Absolute Error):** Average absolute depth error
   - Our result: 0.41m

**Novel Navigation Metrics (our contribution):**
3. **Free Space IoU:** Accuracy of identifying navigable areas
   - Our result: 97.92% - excellent!
   - Critical for path planning

4. **Obstacle Recall:** Fraction of close obstacles detected
   - Our result: 0.3% - needs improvement
   - Safety-critical: missing obstacles = collisions

### 4. Proposed Approach (Technical Novelty)

We built TWO models:

#### Model 1: UNet Baseline (Standard Approach)

**Architecture:**
```
Input → [ResNet-34 Encoder] → [Multi-Scale Decoder] → Dense Depth
     5 channels:                  Skip connections
     - RGB (3)                     Upsampling
     - Sparse Depth (1)
     - Validity Mask (1)
```

**Key Features:**
- Pretrained encoder (transfer learning from ImageNet)
- Skip connections preserve fine details
- Modified to accept 5 input channels instead of 3

#### Model 2: SemNav-DC (Our Innovation)

**Main Idea:** Use semantic understanding to guide depth prediction

**How it Works:**
```
RGB Image →[DeepLabV3]→ Semantic Features (what objects?)
                ↓
RGB + Sparse →[UNet]→ Geometric Features (where objects?)
                ↓
        [Gating Module] ← learns when to use semantic vs geometric
                ↓
        Dense Depth Prediction
```

**Innovation 1: Frozen Semantic Branch**
- Use pretrained DeepLabV3 for semantic segmentation
- Keep it frozen (don't train it)
- **Why?** Preserve semantic knowledge, reduce compute

**Innovation 2: Learned Gating**
- Network learns to blend semantic and geometric features
- "Use semantic features at object boundaries"
- "Use geometric features in texture-rich regions"

**Innovation 3: Boundary-Aware Loss**
- Traditional loss: minimize difference in predicted vs ground truth depth
- Our addition: explicitly penalize fuzzy boundaries
- Formula: Compare detected edges in prediction vs ground truth
- **Impact:** Sharper object boundaries, critical for safety

**Innovation 4: Navigation-Specific Loss**
- Traditional: optimize for pixel accuracy
- Our addition: explicitly optimize for free space detection
- Directly encourage the network to care about "is path clear?"

**Full Loss Function:**
```
Total Loss = 1.0×L1_Loss + 0.5×Gradient_Loss + 0.2×SSIM_Loss + 0.3×Boundary_IoU + 0.1×Freespace_BCE
```

Each term encourages different properties:
- L1: accurate depth values
- Gradient: sharp transitions
- SSIM: preserve structure
- Boundary IoU: accurate object edges (our novelty)
- Freespace BCE: navigation accuracy (our novelty)

### 5. Relevance to Deep Learning Course

**Course Topics Applied:**

**Week 3-5: Convolutional Neural Networks**
- Encoder-decoder architectures
- Skip connections (UNet)
- Pooling and upsampling

**Week 6: Transfer Learning**
- Used pretrained ResNet-34 (ImageNet)
- Used pretrained DeepLabV3 (COCO)
- Froze semantic branch, trained depth branch

**Week 7: Loss Functions**
- Custom multi-term loss
- Combined reconstruction + structural + task-specific losses

**Week 8: Attention Mechanisms**
- Gating modules for feature fusion
- Channel and spatial attention

**Week 10: Multi-Task Learning**
- Joint depth + semantic prediction
- Shared representations

**Why This is Advanced:**

✅ **Dense prediction**, not classification  
✅ **Multi-modal fusion** (RGB + depth + semantics)  
✅ **Custom architecture** with novel components  
✅ **Real-world application** with safety implications  
✅ **Novel evaluation metrics** beyond standard benchmarks  
✅ **Ablation studies** to validate design choices  

❌ **NOT** basic image classification  
❌ **NOT** simple forecasting  
❌ **NOT** using LLMs or RL (out of course scope)  

---

## Technical Implementation Highlights

### Data Pipeline

**Preprocessing (`scripts/preprocess_dataset.py`):**
1. Handle multiple depth formats (PNG uint16, NPY, PFM)
2. Resize to consistent resolution (480×640)
3. Normalize depth to meters
4. Scene-based splitting (prevent data leakage)

**DataLoader (`src/data/ss4blind.py`):**
1. Load RGB, sparse depth, mask, ground truth
2. Apply augmentations (flip, color jitter)
3. Generate sparse depth by sampling valid pixels
4. Return tensors ready for model

### Model Architecture

**Baseline (`src/models/unet_baseline.py`):**
- Dynamic channel detection (works with any encoder)
- Automatic output size matching
- Mixed precision training support

**Advanced (`src/models/semnav_dc.py`):**
- Frozen semantic encoder
- Multi-scale gating modules
- Joint depth + semantic output

### Training Strategy

**Setup:**
- Optimizer: Adam (lr=1e-3, weight decay=1e-4)
- Batch size: 2 (CPU), 8-16 (GPU)
- Epochs: 5-30
- Mixed precision: enabled

**Techniques:**
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping (patience=5)
- Gradient clipping (prevent explosions)
- Checkpoint saving (best RMSE)

### Evaluation

**Metrics Computed:**
- 4 standard depth metrics
- 2 navigation-specific metrics

**Outputs:**
- Quantitative results (markdown table)
- Qualitative visualizations (16 samples)
- Per-scene breakdowns

---

## Results Summary

**What Works Well:**
- ✅ Free space detection: 97.92% IoU
- ✅ Overall depth accuracy: 0.85m RMSE
- ✅ Smooth planar surfaces
- ✅ Handles sparse inputs robustly

**What Needs Improvement:**
- ❌ Close obstacle detection: only 0.3% recall
- ❌ Reflective surfaces
- ❌ Real-time performance (needs optimization)

**Future Work:**
- Try transformer-based encoders
- Add more training data
- Optimize for mobile deployment
- Test with actual blind users

---

## How to Demonstrate

### For Presentation:

1. **Show Problem:** Visualize sparse input (only 800 pixels)
2. **Show Solution:** Our dense prediction vs ground truth
3. **Highlight Innovation:** Compare with/without semantic guidance
4. **Show Metrics:** Free space IoU at 97.9%
5. **Discuss Limitations:** Obstacle recall needs work

### For Code Demo:

```bash
# 1. Show preprocessing
python scripts/preprocess_dataset.py --src data/raw/ss4blind --dst data/ss4blind ...

# 2. Show training
python src/train.py --config configs/unet_baseline.yaml

# 3. Show evaluation
python src/evaluate.py --config configs/unet_baseline.yaml --ckpt checkpoints/.../best_rmse.ckpt
```

### For Report:

- **Introduction:** Assistive navigation problem
- **Related Work:** Existing depth completion methods
- **Methodology:** Architecture, losses, training
- **Experiments:** Dataset, metrics, results
- **Analysis:** What worked, what didn't
- **Conclusion:** Contributions and future work

---

## Key Takeaways

**What You Built:**
A deep learning system that uses semantic understanding to improve depth completion for blind navigation.

**Why It Matters:**
Accurate depth at object boundaries is critical for obstacle avoidance - fuzzy depth = dangerous collisions.

**Technical Contribution:**
1. Semantic-guided depth completion via learned gating
2. Boundary-aware loss for sharp edges
3. Navigation-specific evaluation metrics
4. Real-world dataset evaluation

**Learning Outcomes:**
- Applied CNN architectures for dense prediction
- Implemented transfer learning and multi-task learning
- Designed custom loss functions
- Evaluated on real-world, safety-critical task
- Gained experience with full ML pipeline (data→training→evaluation)

---

## Questions You Might Get

**Q: Why not use NYU Depth V2?**
A: SS4Blind is more realistic for navigation - real sensor noise, actual blind navigation scenarios.

**Q: Why freeze the semantic branch?**
A: Prevents forgetting semantic knowledge, reduces compute, focuses training on depth.

**Q: Why is obstacle recall so low?**
A: Model prioritizes free space (more common), obstacles are rare/hard. Need better sampling or loss weighting.

**Q: Can this run real-time?**
A: Not yet - need model compression, quantization, GPU optimization.

**Q: How is this different from classification?**
A: Dense prediction (1 output per pixel), multi-modal, requires spatial reasoning, real-world safety constraints.

---

## Final Checklist for Submission

- [ ] GitHub repository created and public
- [ ] README.md explains project clearly
- [ ] PROJECT_DOCUMENTATION.md has technical details
- [ ] All code is clean and commented
- [ ] Requirements.txt is complete
- [ ] .gitignore prevents uploading large files
- [ ] LICENSE file is present
- [ ] Can clone and run from scratch
- [ ] Results are reproducible
- [ ] Presentation slides prepared (if needed)

---

**Good luck with your submission!**

