"""
PointNet — 点云分类与分割的统一骨干网络

核心思想：用 max-pooling 聚合无序点云的全局特征（对称函数），解决点云的置换不变性问题。

支持两种模式：
- classification: 输入 N×3 → 输出 C 类
- segmentation: 输入 N×3 → 输出 N×M（每点 M 类）

参考: Qi et al., PointNet: Deep Learning on Point Sets, CVPR 2017
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformNet(nn.Module):
    """T-Net: 学习一个变换矩阵对输入点云做空间对齐"""

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(B, -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        I = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + I
        x = x.view(B, self.k, self.k)
        return x


class PointNetClassifier(nn.Module):
    """PointNet 分类网络"""

    def __init__(self, num_classes=40, num_points=1024):
        super().__init__()
        self.num_points = num_points
        self.input_transform = TransformNet(k=3)
        self.feature_transform = TransformNet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        B, N, _ = x.shape
        x = x.transpose(2, 1)

        t3 = self.input_transform(x)
        x = torch.bmm(x.transpose(2, 1), t3).transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        t64 = self.feature_transform(x)
        x = torch.bmm(x.transpose(2, 1), t64).transpose(2, 1)

        point_feat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        global_feat = torch.max(x, 2, keepdim=True)[0].view(B, -1)

        x = F.relu(self.bn6(self.fc1(global_feat)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, global_feat


class PointNetSegmenter(nn.Module):
    """PointNet 分割网络

    拼接全局特征和局部特征，逐点预测类别。
    """

    def __init__(self, num_classes=40, num_part_classes=6, num_points=1024):
        super().__init__()
        self.num_points = num_points
        self.num_part_classes = num_part_classes
        self.input_transform = TransformNet(k=3)
        self.feature_transform = TransformNet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.convp1 = nn.Conv1d(1088, 512, 1)
        self.convp2 = nn.Conv1d(512, 256, 1)
        self.convp3 = nn.Conv1d(256, 128, 1)
        self.convp4 = nn.Conv1d(128, 128, 1)
        self.convp5 = nn.Conv1d(128, num_part_classes, 1)
        self.bnp1 = nn.BatchNorm1d(512)
        self.bnp2 = nn.BatchNorm1d(256)
        self.bnp3 = nn.BatchNorm1d(128)
        self.bnp4 = nn.BatchNorm1d(128)

    def forward(self, x):
        B, N, _ = x.shape
        x = x.transpose(2, 1)

        t3 = self.input_transform(x)
        x = torch.bmm(x.transpose(2, 1), t3).transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        t64 = self.feature_transform(x)
        x = torch.bmm(x.transpose(2, 1), t64).transpose(2, 1)

        point_feat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        global_feat = torch.max(x, 2, keepdim=True)[0].repeat(1, 1, N)

        combined = torch.cat([point_feat, global_feat], dim=1)

        x = F.relu(self.bnp1(self.convp1(combined)))
        x = F.relu(self.bnp2(self.convp2(x)))
        x = F.relu(self.bnp3(self.convp3(x)))
        x = F.relu(self.bnp4(self.convp4(x)))
        x = self.convp5(x)
        x = x.transpose(2, 1)
        return x


MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]

SHAPENET_PART_CLASSES = {
    0: "Airplane",
    1: "Bag",
    2: "Cap",
    3: "Car",
    4: "Chair",
    5: "Earphone",
    6: "Guitar",
    7: "Knife",
    8: "Lamp",
    9: "Laptop",
    10: "Motorbike",
    11: "Mug",
    12: "Pistol",
    13: "Rocket",
    14: "Skateboard",
    15: "Table",
}

SHAPENET_PART_LABELS = {
    "Airplane": [0, 1, 2, 3],
    "Bag": [4, 5],
    "Cap": [6, 7],
    "Car": [8, 9, 10, 11],
    "Chair": [12, 13, 14, 15],
    "Earphone": [16, 17, 18],
    "Guitar": [19, 20],
    "Knife": [21, 22],
    "Lamp": [23, 24, 25],
    "Laptop": [26, 27],
    "Motorbike": [28, 29, 30, 31, 32],
    "Mug": [33, 34],
    "Pistol": [35, 36],
    "Rocket": [37, 38],
    "Skateboard": [39, 40, 41],
    "Table": [42, 43, 44],
}

PART_NAMES = [
    "wing", "body", "tail", "engine",
    "handle", "wheel",
    "front", "peak",
    "top", "bottom", "wheel", "roof",
    "seat", "back", "leg", "armrest",
    "headphone", "ear", "band",
    "head", "neck",
    "handle", "blade",
    "canopy", "lampshade", "bulb",
    "screen", "keyboard",
    "wheel", "handle", "engine", "motor", "frame",
    "handle", "body",
    "barrel", "trigger",
    "nose", "body", "fin",
    "deck", "wheel", "truck",
    "top", "leg", "shelf",
]
