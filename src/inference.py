"""
推理模块 — 加载模型 + 分类/分割推理

支持：
1. 加载预训练 PointNet 分类模型（ModelNet40）
2. 加载预训练 PointNet 分割模型（ShapeNet Part）
3. 如果没有预训练权重，用随机初始化模型（demo展示用）
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from src.pointnet import (
    PointNetClassifier,
    PointNetSegmenter,
    MODELNET40_CLASSES,
    SHAPENET_PART_CLASSES,
    PART_NAMES,
)


class PointCloudInference:
    def __init__(self, num_points: int = 1024, device: str = "cpu"):
        self.num_points = num_points
        self.device = device
        self.cls_model = None
        self.seg_model = None
        self._load_models()

    def _load_models(self):
        """加载分类和分割模型"""
        print("加载 PointNet 模型...")

        self.cls_model = PointNetClassifier(
            num_classes=len(MODELNET40_CLASSES),
            num_points=self.num_points,
        ).to(self.device)
        self.cls_model.eval()

        self.seg_model = PointNetSegmenter(
            num_part_classes=len(PART_NAMES),
            num_points=self.num_points,
        ).to(self.device)
        self.seg_model.eval()

        cls_weights = Path("models/pointnet_cls.pth")
        seg_weights = Path("models/pointnet_seg.pth")

        if cls_weights.exists():
            state = torch.load(str(cls_weights), map_location=self.device)
            self.cls_model.load_state_dict(state)
            print(f"  分类权重: {cls_weights}")
        else:
            print("  ⚠️ 无分类预训练权重（随机初始化，结果不可靠）")

        if seg_weights.exists():
            state = torch.load(str(seg_weights), map_location=self.device)
            self.seg_model.load_state_dict(state)
            print(f"  分割权重: {seg_weights}")
        else:
            print("  ⚠️ 无分割预训练权重（随机初始化，结果不可靠）")

        print("模型加载完成")

    @torch.no_grad()
    def classify(self, points: np.ndarray, top_k: int = 5) -> Dict[str, float]:
        """点云分类"""
        pts = torch.from_numpy(points).unsqueeze(0).to(self.device)

        logits, _ = self.cls_model(pts)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_indices = np.argsort(probs)[::-1][:top_k]
        results = {}
        for idx in top_indices:
            results[MODELNET40_CLASSES[idx]] = float(probs[idx])
        return results

    @torch.no_grad()
    def segment(self, points: np.ndarray) -> np.ndarray:
        """点云语义分割"""
        pts = torch.from_numpy(points).unsqueeze(0).to(self.device)

        logits = self.seg_model(pts)
        labels = torch.argmax(logits, dim=2).cpu().numpy()[0]
        return labels

    @torch.no_grad()
    def classify_and_segment(self, points: np.ndarray, top_k: int = 5) -> dict:
        """分类 + 分割联合推理"""
        cls_results = self.classify(points, top_k)
        seg_labels = self.segment(points)
        return {
            "classification": cls_results,
            "segmentation": seg_labels,
            "top_class": list(cls_results.keys())[0],
        }
