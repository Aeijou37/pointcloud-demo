"""
训练脚本 — PointNet 分类与分割训练

支持在 ModelNet40 / ShapeNet Part 上训练。
回家后如果有预训练权重就不需要训练，直接用 demo。

运行:
  python src/train.py --mode classification --dataset ModelNet40 --epochs 50
  python src/train.py --mode segmentation --dataset ShapeNetPart --epochs 100
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from src.pointnet import (
    PointNetClassifier,
    PointNetSegmenter,
    MODELNET40_CLASSES,
    PART_NAMES,
)


def get_dataset(dataset_name: str, mode: str, num_points: int):
    """获取数据集"""
    if dataset_name == "ModelNet40":
        return get_modelnet40(num_points, mode)
    elif dataset_name == "ShapeNetPart":
        return get_shapenet_part(num_points, mode)
    else:
        raise ValueError(f"未知数据集: {dataset_name}")


def get_modelnet40(num_points: int, mode: str):
    """ModelNet40 数据集

    下载: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
    放到 ./data/modelnet40_normal_resampled/
    """
    from torch.utils.data import Dataset

    class ModelNet40Dataset(Dataset):
        def __init__(self, root="./data/modelnet40_normal_resampled", num_points=1024, split="train"):
            self.root = Path(root)
            self.num_points = num_points
            self.catfile = self.root / "modelnet40_shape_names.txt"
            self.cat = [l.strip() for l in open(self.catfile)]
            self.classes = dict(zip(self.cat, range(len(self.cat))))

            split_file = self.root / f"modelnet40_{split}.txt"
            with open(split_file) as f:
                self.file_list = [l.strip() for l in f]

            self.datapath = []
            for line in self.file_list:
                cls_name = line.split("_")[0]
                if cls_name not in self.classes:
                    cls_name = line.rsplit("_", 1)[0]
                file_path = self.root / (line + ".txt")
                if file_path.exists():
                    self.datapath.append((file_path, self.classes[cls_name]))

            print(f"ModelNet40 {split}: {len(self.datapath)} samples")

        def __len__(self):
            return len(self.datapath)

        def __getitem__(self, idx):
            file_path, label = self.datapath[idx]
            points = np.loadtxt(str(file_path), delimiter=",")[:, :3]
            from src.utils import sample_points, normalize_points
            points = sample_points(points, self.num_points)
            points = normalize_points(points)
            return torch.from_numpy(points).float(), label

    train_ds = ModelNet40Dataset(split="train", num_points=num_points)
    test_ds = ModelNet40Dataset(split="test", num_points=num_points)
    return train_ds, test_ds, len(MODELNET40_CLASSES)


def get_shapenet_part(num_points: int, mode: str):
    """ShapeNet Part 数据集

    下载: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip
    放到 ./data/shapenetpart/
    """
    from torch.utils.data import Dataset

    class ShapeNetPartDataset(Dataset):
        def __init__(self, root="./data/shapenetpart", num_points=1024, split="train"):
            self.root = Path(root)
            self.num_points = num_points
            self.catfile = self.root / "synsetoffset2category.txt"
            self.cat = {}
            with open(self.catfile) as f:
                for line in f:
                    ls = line.strip().split()
                    self.cat[ls[0]] = ls[1]

            self.classes = {cat: i for i, cat in enumerate(self.cat)}
            self.seg_classes = {}

            import glob
            all_files = glob.glob(str(self.root / "*" / "*" / "*.txt"))
            split_file = self.root / "train_test_split" / f"shuffled_{split}_file_list.json"
            if split_file.exists():
                import json
                with open(split_file) as f:
                    split_ids = set(json.load(f))
                all_files = [f for f in all_files if "/".join(Path(f).parts[-2:]) in split_ids]

            self.datapath = []
            for f in all_files:
                cat_id = Path(f).parts[-3]
                for name, cid in self.cat.items():
                    if cid == cat_id:
                        self.datapath.append((f, self.classes[name]))
                        break

            print(f"ShapeNetPart {split}: {len(self.datapath)} samples")

        def __len__(self):
            return len(self.datapath)

        def __getitem__(self, idx):
            file_path, label = self.datapath[idx]
            data = np.loadtxt(str(file_path))[:, :6]
            points = data[:, :3]
            seg = data[:, -1].astype(int)

            from src.utils import sample_points
            indices = np.random.choice(len(points), self.num_points, replace=len(points) < self.num_points)
            points = points[indices]
            seg = seg[indices]

            from src.utils import normalize_points
            points = normalize_points(points)
            return torch.from_numpy(points).float(), label, torch.from_numpy(seg).long()

    train_ds = ShapeNetPartDataset(split="trainval", num_points=num_points)
    test_ds = ShapeNetPartDataset(split="test", num_points=num_points)
    return train_ds, test_ds, len(PART_NAMES)


def train_classification(args):
    """训练分类模型"""
    from torch.utils.data import DataLoader

    device = torch.device(args.device)
    train_ds, test_ds, num_classes = get_dataset(args.dataset, "classification", args.num_points)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = PointNetClassifier(num_classes=num_classes, num_points=args.num_points).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    print(f"\n训练分类: {args.dataset}, {len(train_ds)} samples, {args.epochs} epochs")
    print("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (points, labels) in enumerate(train_loader):
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        acc = correct / total
        print(f"E{epoch+1}/{args.epochs} | loss={total_loss/len(train_loader):.4f} | acc={acc:.4f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/pointnet_cls.pth")
    print("\n分类模型已保存: models/pointnet_cls.pth")


def train_segmentation(args):
    """训练分割模型"""
    from torch.utils.data import DataLoader

    device = torch.device(args.device)
    train_ds, test_ds, num_part_classes = get_dataset(args.dataset, "segmentation", args.num_points)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = PointNetSegmenter(num_part_classes=num_part_classes, num_points=args.num_points).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n训练分割: {args.dataset}, {len(train_ds)} samples, {args.epochs} epochs")
    print("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (points, labels, seg) in enumerate(train_loader):
            points, seg = points.to(device), seg.to(device)
            optimizer.zero_grad()
            logits = model(points)
            loss = criterion(logits.view(-1, num_part_classes), seg.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=2)
            correct += (preds == seg).sum().item()
            total += seg.numel()

        acc = correct / total
        print(f"E{epoch+1}/{args.epochs} | loss={total_loss/len(train_loader):.4f} | acc={acc:.4f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/pointnet_seg.pth")
    print("\n分割模型已保存: models/pointnet_seg.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointNet 训练")
    parser.add_argument("--mode", type=str, required=True, choices=["classification", "segmentation"])
    parser.add_argument("--dataset", type=str, default="ModelNet40")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.mode == "classification":
        train_classification(args)
    else:
        train_segmentation(args)
