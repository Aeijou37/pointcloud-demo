---
title: 3D Point Cloud Demo
emoji: ☁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
app_file: src/app.py
pinned: false
---

# 3D Point Cloud Demo

> PointNet-based 3D point cloud classification and segmentation demo.
> Upload a point cloud or pick a preset shape → get classification + part segmentation with 3D visualization.

---

## 📌 Overview

This project demonstrates **PointNet** for 3D point cloud understanding — both **classification** (what object is this?) and **semantic segmentation** (which part does each point belong to?).

It complements my industrial shoe-sole project (which involves 3D reconstruction and point cloud trajectory extraction but is under NDA) with a **publicly demonstrable** 3D vision capability.

---

## ✨ Features

- **Classification**: PointNet on ModelNet40 (40 object classes), Top-5 predictions
- **Segmentation**: PointNet on ShapeNet Part (16 object categories, 50 parts), per-point labels
- **3D Visualization**: Matplotlib 3D rendering with rotatable view (elevation/azimuth)
- **Preset shapes**: sphere, cube, cylinder, cone, chair, table, airplane
- **File upload**: .ply / .pcd / .txt / .npy / .xyz
- **Gradio web interface**: interactive demo, no code needed

---

## 🧠 Technical Stack

| Component | Selection |
|---|---|
| Model | PointNet (Qi et al., CVPR 2017) |
| Classification | ModelNet40 (40 classes) |
| Segmentation | ShapeNet Part (50 parts) |
| Visualization | Matplotlib 3D + Open3D |
| Frontend | Gradio |
| Training | PyTorch |

---

## 🚀 Quick Start

### Run locally

```bash
pip install -r requirements.txt
python src/app.py
```

Open `http://localhost:7860`. Select a preset shape (e.g., "chair") and click "运行".

### Run with GPU

```bash
python src/app.py --device cuda
```

### Train your own models (optional)

```bash
# Download ModelNet40
# https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
# Extract to ./data/modelnet40_normal_resampled/

# Train classification
python src/train.py --mode classification --dataset ModelNet40 --epochs 50 --device cuda

# Download ShapeNet Part
# https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip
# Extract to ./data/shapenetpart/

# Train segmentation
python src/train.py --mode segmentation --dataset ShapeNetPart --epochs 100 --device cuda
```

Trained weights are saved to `models/pointnet_cls.pth` and `models/pointnet_seg.pth`.

---

## 📁 Project Structure

```
pointcloud-demo/
├── README.md
├── requirements.txt
└── src/
    ├── pointnet.py      # PointNet model (classifier + segmenter + T-Net)
    ├── inference.py     # Inference module (load model + predict)
    ├── utils.py         # Point cloud utils (load/sample/normalize/visualize)
    ├── train.py         # Training script (ModelNet40 / ShapeNet Part)
    └── app.py           # Gradio frontend
```

---

## 📊 Key Design Decisions

### 1. PointNet for permutation invariance

Point clouds are unordered sets — feeding them into a standard CNN treats the point order as meaningful, which it isn't. PointNet uses **max-pooling** as a symmetric function to aggregate global features, making the output invariant to input ordering.

### 2. T-Net for spatial alignment

Input and feature transform networks (T-Net) learn rotation matrices to align the point cloud before feature extraction, improving robustness to pose variations.

### 3. Classification → Segmentation via feature fusion

Segmentation concatenates **local point features** (64-dim) with **global features** (1024-dim) for each point, giving both local context and global context for per-point prediction.

---

## 👤 Author

**Guojie Li**

GitHub: [Aeijou37](https://github.com/Aeijou37)
