"""
点云处理工具 — 采样、归一化、可视化

职责：
1. 加载点云文件（ply/pcd/txt/npy）
2. 随机采样到固定点数
3. 归一化到单位球
4. 3D可视化 → 2D图像（用于Gradio展示）
5. 生成预设示例点云（球/立方体/椅子简化版）
"""
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_pointcloud(file_path: str, num_points: int = 1024) -> np.ndarray:
    """加载点云文件，返回 N×3 numpy 数组"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        points = np.load(file_path)
    elif suffix in [".ply", ".pcd"]:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
    elif suffix in [".txt", ".xyz", ".csv"]:
        points = np.loadtxt(file_path, delimiter=",") if suffix == ".csv" else np.loadtxt(file_path)
        if points.shape[1] > 3:
            points = points[:, :3]
    else:
        raise ValueError(f"不支持的格式: {suffix}")

    points = sample_points(points, num_points)
    points = normalize_points(points)
    return points.astype(np.float32)


def sample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """随机采样/重复采样到固定点数"""
    N = points.shape[0]
    if N == 0:
        raise ValueError("点云为空")
    if N >= num_points:
        indices = np.random.choice(N, num_points, replace=False)
    else:
        indices = np.random.choice(N, num_points, replace=True)
    return points[indices]


def normalize_points(points: np.ndarray) -> np.ndarray:
    """归一化到单位球（中心化 + 缩放）"""
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
    return points


def visualize_pointcloud(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Point Cloud",
    figsize: Tuple[int, int] = (8, 8),
    elev: int = 30,
    azim: int = 45,
) -> str:
    """3D可视化点云，保存为图像，返回路径"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_labels), 1)))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                points[mask, 0], points[mask, 1], points[mask, 2],
                c=[colors[i % len(colors)]], s=1.5, label=f"Part {label}", alpha=0.8
            )
        ax.legend(loc="upper right", fontsize=8, markerscale=4)
    else:
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c="steelblue", s=1.5, alpha=0.8
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    max_range = max(
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min(),
    ) / 2
    mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
    mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
    mid_z = (points[:, 2].max() + points[:, 2].min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.view_init(elev=elev, azim=azim)

    output_path = "output_pointcloud.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def visualize_topk_classification(
    points: np.ndarray,
    predictions: dict,
    title: str = "Classification Results",
) -> str:
    """可视化Top-K分类结果"""
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c="steelblue", s=1.5, alpha=0.8)
    ax1.set_title("Input Point Cloud", fontsize=10)
    mid = points.mean(axis=0)
    r = max(
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min(),
    ) / 2
    ax1.set_xlim(mid[0]-r, mid[0]+r)
    ax1.set_ylim(mid[1]-r, mid[1]+r)
    ax1.set_zlim(mid[2]-r, mid[2]+r)
    ax1.view_init(elev=30, azim=45)

    ax2 = fig.add_subplot(122)
    classes = list(predictions.keys())
    probs = list(predictions.values())
    colors = ["#2196F3" if i == 0 else "#90CAF9" for i in range(len(classes))]
    bars = ax2.barh(range(len(classes)), probs, color=colors)
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(classes, fontsize=10)
    ax2.set_xlabel("Probability")
    ax2.set_title("Top-5 Predictions", fontsize=10)
    ax2.invert_yaxis()
    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{prob:.1%}", va="center", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    output_path = "output_classification.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def generate_shape(shape: str, num_points: int = 1024) -> np.ndarray:
    """生成预设形状的点云（用于无文件时的demo）"""
    if shape == "sphere":
        phi = np.random.uniform(0, np.pi, num_points)
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points = np.stack([x, y, z], axis=1)
    elif shape == "cube":
        points = np.random.uniform(-1, 1, (num_points, 3))
    elif shape == "cylinder":
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        r = np.ones(num_points) * 0.8
        z = np.random.uniform(-1, 1, num_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.stack([x, y, z], axis=1)
    elif shape == "cone":
        z = np.random.uniform(0, 1.5, num_points)
        r = 1.0 - z / 1.5
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.stack([x, y, z - 0.75], axis=1)
    elif shape == "chair":
        seat = np.random.uniform(-0.5, 0.5, (num_points // 4, 3))
        seat[:, 2] = 0.5
        back = np.random.uniform(-0.5, 0.5, (num_points // 4, 3))
        back[:, 1] = 0.5
        back[:, 2] = np.random.uniform(0.5, 1.5, num_points // 4)
        leg1 = np.random.uniform(-0.05, 0.05, (num_points // 8, 3))
        leg1[:, 0] -= 0.45; leg1[:, 1] -= 0.45; leg1[:, 2] = np.random.uniform(0, 0.5, num_points // 8)
        leg2 = np.random.uniform(-0.05, 0.05, (num_points // 8, 3))
        leg2[:, 0] += 0.45; leg2[:, 1] -= 0.45; leg2[:, 2] = np.random.uniform(0, 0.5, num_points // 8)
        leg3 = np.random.uniform(-0.05, 0.05, (num_points // 8, 3))
        leg3[:, 0] -= 0.45; leg3[:, 1] += 0.45; leg3[:, 2] = np.random.uniform(0, 0.5, num_points // 8)
        leg4 = np.random.uniform(-0.05, 0.05, (num_points // 8, 3))
        leg4[:, 0] += 0.45; leg4[:, 1] += 0.45; leg4[:, 2] = np.random.uniform(0, 0.5, num_points // 8)
        points = np.vstack([seat, back, leg1, leg2, leg3, leg4])
        points = sample_points(points, num_points)
    elif shape == "table":
        top = np.random.uniform(-0.6, 0.6, (num_points // 3, 3))
        top[:, 2] = 0.7
        leg1 = np.random.uniform(-0.05, 0.05, (num_points // 6, 3))
        leg1[:, 0] -= 0.5; leg1[:, 1] -= 0.5; leg1[:, 2] = np.random.uniform(0, 0.7, num_points // 6)
        leg2 = np.random.uniform(-0.05, 0.05, (num_points // 6, 3))
        leg2[:, 0] += 0.5; leg2[:, 1] -= 0.5; leg2[:, 2] = np.random.uniform(0, 0.7, num_points // 6)
        leg3 = np.random.uniform(-0.05, 0.05, (num_points // 6, 3))
        leg3[:, 0] -= 0.5; leg3[:, 1] += 0.5; leg3[:, 2] = np.random.uniform(0, 0.7, num_points // 6)
        leg4 = np.random.uniform(-0.05, 0.05, (num_points // 6, 3))
        leg4[:, 0] += 0.5; leg4[:, 1] += 0.5; leg4[:, 2] = np.random.uniform(0, 0.7, num_points // 6)
        points = np.vstack([top, leg1, leg2, leg3, leg4])
        points = sample_points(points, num_points)
    elif shape == "airplane":
        fuselage = np.random.uniform(-0.1, 0.1, (num_points // 2, 3))
        fuselage[:, 0] = np.random.uniform(-1.0, 1.0, num_points // 2)
        wing = np.random.uniform(-0.1, 0.1, (num_points // 4, 3))
        wing[:, 1] = np.random.uniform(-1.0, 1.0, num_points // 4)
        wing[:, 0] = np.random.uniform(-0.2, 0.2, num_points // 4)
        tail = np.random.uniform(-0.1, 0.1, (num_points // 4, 3))
        tail[:, 0] = np.random.uniform(0.8, 1.0, num_points // 4)
        tail[:, 2] = np.random.uniform(0, 0.4, num_points // 4)
        points = np.vstack([fuselage, wing, tail])
        points = sample_points(points, num_points)
    else:
        points = np.random.uniform(-1, 1, (num_points, 3))

    points = normalize_points(points)
    return points.astype(np.float32)


SHAPES = ["sphere", "cube", "cylinder", "cone", "chair", "table", "airplane"]
