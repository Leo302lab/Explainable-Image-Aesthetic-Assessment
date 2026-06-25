# /root/autodl-tmp/pythonProject2/baseline/TAVAR/TAVAR_training.py
# TAVAR Baseline Training Script - Fixed Version
# 关键修复：
# 1) 去掉回归头中的 BatchNorm1d，避免 batch size=8 时 train/eval 不一致
# 2) 预训练主干只加载 backbone，不加载随机/不匹配 FC 头
# 3) backbone 与 regressor 使用分组学习率
# 4) weight_decay 从 0.05 改为 1e-4，并排除 bias/norm 参数
# 5) 前几个 epoch 冻结 backbone，只训练随机初始化的 regressor
# 6) 增加 label 检查、NaN 清理、overfit debug 模式

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torch.amp import autocast, GradScaler


# ===================== Configuration =====================
PROJECT_ROOT = "/root/autodl-tmp/pythonProject2"
sys.path.insert(0, PROJECT_ROOT)

cfg = None
try:
    from utils.config import cfg
except ImportError:
    pass

if cfg is not None:
    IMAGE_DIR = getattr(cfg, "IMAGE_DIR", "/root/autodl-tmp/Data_preprocess/cleaned")
    FEATURE_DIR = getattr(
        cfg,
        "FEATURE_DIR",
        "/root/autodl-tmp/Data_preprocess/features_Fixed_final_design_roi_hardmask",
    )
else:
    IMAGE_DIR = "/root/autodl-tmp/Data_preprocess/cleaned"
    FEATURE_DIR = "/root/autodl-tmp/Data_preprocess/features_Fixed_final_design_roi_hardmask"

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/baseline")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PRETRAINED_WEIGHTS = os.path.join(PROJECT_ROOT, "baseline/TAVAR/TAVAR_weight.pt")

# Training hyperparameters
EPOCHS = 30
BATCH_SIZE = 8
ACCUMULATION_STEPS = 1
GRAD_CLIP = 5.0

# 关键修改：随机初始化的 regressor 要比 backbone 学得快
LR_BACKBONE = 1e-5
LR_HEAD = 3e-4

# 关键修改：ResNet 回归 baseline 不建议用 0.05
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
EARLY_STOP_PATIENCE = 10

# 前几个 epoch 冻结 backbone，只训练随机初始化的回归头
FREEZE_BACKBONE_EPOCHS = 3

# 小 batch 微调时，建议固定 ResNet 的 BatchNorm2d running stats
FREEZE_BACKBONE_BN = True

# 与论文中多种子保持一致；如果你确实要用 2026，可以自行改回
SUPPORTED_SEEDS = [42, 123, 2024]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if cfg is not None:
    IMAGE_SIZE = int(getattr(cfg, "IMAGE_SIZE", 224))
    MEAN = list(getattr(cfg, "MEAN", [0.485, 0.456, 0.406]))
    STD = list(getattr(cfg, "STD", [0.229, 0.224, 0.225]))
else:
    IMAGE_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]


# ===================== Dataset =====================
def _normalize_image_id_string(x) -> str:
    s = str(x).strip()
    if s.lower().endswith((".jpg", ".png", ".jpeg")):
        s = os.path.splitext(s)[0]
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _resolve_image_path(image_dir: str, img_id: str):
    s = _normalize_image_id_string(img_id)
    candidates = [s]
    if s.isdigit():
        candidates.append(str(int(s)))
        candidates.append(s.zfill(5))
        candidates.append(s.zfill(6))

    seen = set()
    for cid in candidates:
        if cid in seen:
            continue
        seen.add(cid)
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            p = os.path.join(image_dir, f"{cid}{ext}")
            if os.path.exists(p):
                return p
    return None


class TAVARDataset(Dataset):
    """AVA 数据集加载器，用于 TAVAR/ResNet baseline。"""

    def __init__(self, split, images_dir, feature_dir, is_train=True):
        csv_path = os.path.join(feature_dir, f"{split}_features.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 找不到特征文件: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.split = split
        self.transform = self._build_transform(is_train)

        label_col = None
        if cfg is not None and hasattr(cfg, "LABEL_COLUMN") and cfg.LABEL_COLUMN in self.df.columns:
            label_col = cfg.LABEL_COLUMN
        elif "label" in self.df.columns:
            label_col = "label"

        if label_col is not None:
            self.labels = pd.to_numeric(self.df[label_col], errors="coerce").values.astype(np.float32)
            used_label = label_col
        elif "mean_score" in self.df.columns:
            print(f"⚠️ [{split}] 未找到归一化 label，使用 mean_score 并归一化到 [0,1]")
            mean_score = pd.to_numeric(self.df["mean_score"], errors="coerce").values.astype(np.float32)
            self.labels = (mean_score - 1.0) / 9.0
            used_label = "mean_score -> normalized"
        else:
            raise KeyError(f"❌ [{split}] 找不到训练标签列。columns={list(self.df.columns)}")

        # 清理 NaN 标签
        nan_mask = np.isnan(self.labels)
        if nan_mask.any():
            print(f"⚠️ [{split}] 发现 NaN 标签 {nan_mask.sum()} 个，已删除对应样本")
            self.df = self.df.loc[~nan_mask].reset_index(drop=True)
            self.labels = self.labels[~nan_mask]

        valid = self.labels
        unique_count = len(np.unique(np.round(valid, 6)))

        print(
            f"   [{split}] label_col={used_label} | "
            f"n={len(self.labels)}, min={valid.min():.4f}, max={valid.max():.4f}, "
            f"mean={valid.mean():.4f}, std={valid.std():.4f}, unique≈{unique_count}"
        )

        if unique_count <= 5:
            print(
                f"   ⚠️ [{split}] label unique 很少，可能是二分类标签。"
                f"如果是 0/1 标签，不适合用当前 MSE 回归 + SRCC 方案。"
            )

        if valid.min() < -0.05 or valid.max() > 1.05:
            print(
                f"   ⚠️ [{split}] label 看起来不在 [0,1] 范围内。"
                f"请确认 cfg.LABEL_COLUMN 是否指向了正确的归一化连续美学分数。"
            )

    def _build_transform(self, is_train):
        if is_train:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id_val = row["image_id"] if "image_id" in self.df.columns else row.iloc[0]
        image_path = _resolve_image_path(self.images_dir, str(img_id_val))
        if image_path is None:
            raise FileNotFoundError(f"❌ 找不到图像: {img_id_val} in {self.images_dir}")

        image = Image.open(image_path).convert("RGB")
        x = self.transform(image)
        label = float(self.labels[idx])

        return x, np.array([label], dtype=np.float32)


# ===================== TAVAR / ResNet Model =====================
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class TAVAR_ResNet(nn.Module):
    """TAVAR 权重主干 + 简洁回归头。

    注意：
    - 主干保持 ResNet50 结构，以兼容 TAVAR_weight.pt 的 AttrNet.resnet.* 权重。
    - 回归头不使用 BatchNorm1d，避免小 batch 下 train/eval 不一致。
    - 输出为线性回归值，不加 Sigmoid。
    """

    def __init__(self, block=Bottleneck, layers=(3, 4, 6, 3), num_classes=1):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 关键修复：不使用 BatchNorm1d
        self.regressor = nn.Sequential(
            nn.Linear(512 * block.expansion, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 最后一层回归输出初始化得更温和一些
        final_fc = self.regressor[-1]
        if isinstance(final_fc, nn.Linear):
            nn.init.normal_(final_fc.weight, mean=0.0, std=0.01)
            nn.init.constant_(final_fc.bias, 0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        out = self.regressor(x)
        return out


# ===================== Pretrained Weight Loading =====================
def clean_pretrained_key(k: str) -> str:
    """递归清理 checkpoint key 前缀。"""
    prefixes = [
        "module.",
        "AttrNet.resnet.",
        "AttrNet.",
        "resnet.",
        "backbone.",
        "encoder.",
        "features.",
    ]

    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                changed = True
                break
    return k


def build_tavar_model(pretrained=True, verbose=True):
    model = TAVAR_ResNet()

    pretrained_dict = None
    if pretrained and os.path.exists(PRETRAINED_WEIGHTS):
        try:
            checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                pretrained_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model" in checkpoint:
                pretrained_dict = checkpoint["model"]
            else:
                pretrained_dict = checkpoint
        except Exception as e:
            print(f"⚠️ 加载预训练权重失败: {e}")
            pretrained_dict = None

    if pretrained_dict is None:
        if pretrained and verbose:
            print(f"⚠️ 预训练权重不存在或读取失败: {PRETRAINED_WEIGHTS}")
            print("   将使用随机初始化")
        return model

    cleaned = {}
    for k, v in pretrained_dict.items():
        cleaned[clean_pretrained_key(k)] = v
    pretrained_dict = cleaned

    model_dict = model.state_dict()

    backbone_keys = {
        k for k in model_dict.keys()
        if k.startswith("conv1.")
        or k.startswith("bn1.")
        or k.startswith("layer1.")
        or k.startswith("layer2.")
        or k.startswith("layer3.")
        or k.startswith("layer4.")
    }

    # 只加载 backbone；regressor 始终随机初始化
    matched_backbone = {
        k: pretrained_dict[k]
        for k in backbone_keys
        if k in pretrained_dict and pretrained_dict[k].shape == model_dict[k].shape
    }

    missing_backbone = backbone_keys - set(matched_backbone.keys())

    updated = dict(model_dict)
    updated.update(matched_backbone)
    model.load_state_dict(updated, strict=True)

    if verbose:
        regressor_keys = [k for k in model_dict.keys() if k.startswith("regressor.")]
        print(f"✅ [TAVAR] 预训练权重: {PRETRAINED_WEIGHTS}")
        print(f"   主干匹配: {len(matched_backbone)}/{len(backbone_keys)} keys")
        if missing_backbone:
            print(f"   ⚠️ 主干缺失 ({len(missing_backbone)}): {sorted(list(missing_backbone))[:5]}...")
        print(f"   Regressor: {len(regressor_keys)} keys，全部随机初始化")
        print(f"   预训练总keys: {len(pretrained_dict)}")

    return model


# ===================== Training Utils =====================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, deterministic=False, verbose=True):
    if verbose:
        print("\n" + "=" * 50)
        print(f"🎲 设置随机种子: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if verbose:
        print("✅ 种子已设置")
        print("=" * 50)


def pearson_corr_np(pred, target):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    pred_std = pred.std()
    target_std = target.std()
    if pred_std < 1e-12 or target_std < 1e-12:
        return 0.0

    pred_mean = pred.mean()
    target_mean = target.mean()
    cov = ((pred - pred_mean) * (target - target_mean)).mean()
    return float(cov / (pred_std * target_std + 1e-8))


def compute_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)

    # 指标使用反归一化后的 1-10 分数；SRCC/PCC 不受正向线性缩放影响
    pred_score = predictions * 9.0 + 1.0
    target_score = targets * 9.0 + 1.0

    mse = float(np.mean((pred_score - target_score) ** 2))
    mae = float(np.mean(np.abs(pred_score - target_score)))

    srcc = stats.spearmanr(pred_score, target_score)[0]
    if np.isnan(srcc):
        srcc = 0.0

    plcc = pearson_corr_np(pred_score, target_score)

    # 额外输出预测范围，方便判断是否崩掉
    return {
        "srcc": float(srcc),
        "plcc": float(plcc),
        "mae": mae,
        "mse": mse,
        "pred_min": float(predictions.min()),
        "pred_max": float(predictions.max()),
        "pred_mean": float(predictions.mean()),
        "target_min": float(targets.min()),
        "target_max": float(targets.max()),
        "target_mean": float(targets.mean()),
    }


def set_backbone_trainable(model, trainable: bool):
    for name, param in model.named_parameters():
        if not name.startswith("regressor."):
            param.requires_grad = trainable


def set_batchnorm2d_eval(model):
    """小 batch 微调时固定 backbone BN running stats。"""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def make_optimizer(model):
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_head = name.startswith("regressor.")
        is_no_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "bn" in name.lower()
            or "norm" in name.lower()
        )

        if is_head and is_no_decay:
            head_no_decay.append(param)
        elif is_head:
            head_decay.append(param)
        elif is_no_decay:
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)

    param_groups = []
    if backbone_decay:
        param_groups.append({"params": backbone_decay, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY})
    if backbone_no_decay:
        param_groups.append({"params": backbone_no_decay, "lr": LR_BACKBONE, "weight_decay": 0.0})
    if head_decay:
        param_groups.append({"params": head_decay, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY})
    if head_no_decay:
        param_groups.append({"params": head_no_decay, "lr": LR_HEAD, "weight_decay": 0.0})

    return optim.AdamW(param_groups)


def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, amp_enabled=True):
    model.train()

    if FREEZE_BACKBONE_BN:
        set_batchnorm2d_eval(model)

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Ep {epoch} [Train]", leave=False)
    optimizer.zero_grad(set_to_none=True)

    accumulation_steps = max(ACCUMULATION_STEPS, 1)

    for step, (images, targets) in enumerate(pbar):
        images = images.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
        targets = targets.to(DEVICE, non_blocking=(DEVICE.type == "cuda")).float().view(-1)

        with autocast(device_type=DEVICE.type, enabled=amp_enabled):
            outputs = model(images).view(-1)
            loss = criterion(outputs, targets) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size * accumulation_steps
        total_samples += batch_size

        all_preds.append(outputs.detach().float().cpu().numpy())
        all_targets.append(targets.detach().float().cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_metrics(all_preds, all_targets)

    print(
        f"📊 [Train] Ep {epoch}: "
        f"SRCC={metrics['srcc']:.4f} | PCC={metrics['plcc']:.4f} | "
        f"MAE={metrics['mae']:.4f} | MSE={metrics['mse']:.4f} | Loss={avg_loss:.4f} | "
        f"pred=[{metrics['pred_min']:.3f},{metrics['pred_max']:.3f}], mean={metrics['pred_mean']:.3f}"
    )

    return avg_loss, metrics


@torch.no_grad()
def validate(model, loader, criterion, epoch, amp_enabled=True, desc="[Val ]"):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(loader, desc=f"Ep {epoch} {desc}", leave=False):
        images = images.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
        targets = targets.to(DEVICE, non_blocking=(DEVICE.type == "cuda")).float().view(-1)

        with autocast(device_type=DEVICE.type, enabled=amp_enabled):
            outputs = model(images).view(-1)
            loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.append(outputs.detach().float().cpu().numpy())
        all_targets.append(targets.detach().float().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_metrics(all_preds, all_targets)

    print(
        f"✨ {desc} Ep {epoch}: "
        f"SRCC={metrics['srcc']:.4f} | PCC={metrics['plcc']:.4f} | "
        f"MAE={metrics['mae']:.4f} | MSE={metrics['mse']:.4f} | Loss={avg_loss:.4f} | "
        f"pred=[{metrics['pred_min']:.3f},{metrics['pred_max']:.3f}], mean={metrics['pred_mean']:.3f}"
    )

    return avg_loss, metrics


# ===================== Main =====================
def build_dataloaders(seed, overfit_debug=False, overfit_samples=256):
    print("\n📦 Initializing Image Datasets...")

    # overfit debug 时关闭训练增强，确保能快速验证图像-标签-训练链路
    train_is_train = not overfit_debug

    train_dataset = TAVARDataset(
        split="train",
        images_dir=IMAGE_DIR,
        feature_dir=FEATURE_DIR,
        is_train=train_is_train,
    )

    val_dataset = TAVARDataset(
        split="val",
        images_dir=IMAGE_DIR,
        feature_dir=FEATURE_DIR,
        is_train=False,
    )

    test_dataset = TAVARDataset(
        split="test",
        images_dir=IMAGE_DIR,
        feature_dir=FEATURE_DIR,
        is_train=False,
    )

    if overfit_debug:
        n = min(overfit_samples, len(train_dataset))
        indices = list(range(n))
        train_dataset = Subset(train_dataset, indices)

        # 过拟合诊断：验证集也用同一批训练样本
        val_dataset = train_dataset
        test_dataset = train_dataset
        print(f"🧪 Overfit Debug Mode: train/val/test 均使用 train 前 {n} 个样本")

    num_workers = 4 if torch.cuda.is_available() and not overfit_debug else 0
    g = torch.Generator()
    g.manual_seed(seed)

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": num_workers,
        "pin_memory": bool(DEVICE.type == "cuda"),
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = seed_worker

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=g,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        generator=g,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        generator=g,
        **loader_kwargs,
    )

    print(f"   ✅ Train Set Size: {len(train_dataset)}")
    print(f"   ✅ Val Set Size:   {len(val_dataset)}")
    print(f"   ✅ Test Set Size:  {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def main(seed=42, overfit_debug=False, overfit_samples=256):
    amp_enabled = DEVICE.type == "cuda"

    print(f"\n{'=' * 60}")
    print(f"🎲 训练 TAVAR Baseline Fixed (seed={seed})")
    print("=" * 60)

    print(f"🖥️ [Device] Using device: {DEVICE}")
    set_seed(seed)

    train_loader, val_loader, test_loader = build_dataloaders(
        seed=seed,
        overfit_debug=overfit_debug,
        overfit_samples=overfit_samples,
    )

    print("\n🏗️ Building TAVAR Model...")
    model = build_tavar_model(pretrained=True).to(DEVICE)
    print("✅ Built TAVAR model")

    if FREEZE_BACKBONE_EPOCHS > 0 and not overfit_debug:
        set_backbone_trainable(model, trainable=False)
        print(f"🧊 Backbone frozen for first {FREEZE_BACKBONE_EPOCHS} epoch(s)")
    else:
        set_backbone_trainable(model, trainable=True)

    criterion = nn.MSELoss()
    optimizer = make_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-7,
    )
    scaler = GradScaler(DEVICE.type, enabled=amp_enabled)

    best_srcc = -1.0
    best_epoch = -1
    early_stop_counter = 0

    suffix = "overfit_debug" if overfit_debug else f"seed{seed}"
    save_path = os.path.join(CHECKPOINT_DIR, f"tavar_fixed_{suffix}.pth")

    print(
        f"\n🚀 Start Training "
        f"(Epochs={EPOCHS}, ImgSize={IMAGE_SIZE}, Batch={BATCH_SIZE}, "
        f"Accum={ACCUMULATION_STEPS}, LR_backbone={LR_BACKBONE}, LR_head={LR_HEAD}, "
        f"WD={WEIGHT_DECAY}, Drop={DROPOUT}, FreezeEpochs={FREEZE_BACKBONE_EPOCHS}, "
        f"FreezeBN={FREEZE_BACKBONE_BN}, Clip={GRAD_CLIP}, AMP={amp_enabled})..."
    )

    for epoch in range(1, EPOCHS + 1):
        if (
            FREEZE_BACKBONE_EPOCHS > 0
            and not overfit_debug
            and epoch == FREEZE_BACKBONE_EPOCHS + 1
        ):
            set_backbone_trainable(model, trainable=True)
            optimizer = make_optimizer(model)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(EPOCHS - epoch + 1, 1),
                eta_min=1e-7,
            )
            print(f"🔥 Epoch {epoch}: Backbone unfrozen. Rebuilt optimizer.")

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            epoch,
            amp_enabled,
        )

        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            epoch,
            amp_enabled,
            desc="[Val ]",
        )

        scheduler.step()

        if val_metrics["srcc"] > best_srcc + 1e-4:
            best_srcc = val_metrics["srcc"]
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "best_srcc": best_srcc,
                    "val_metrics": val_metrics,
                },
                save_path,
            )
            print(f"🏆 New Best Saved: SRCC={best_srcc:.4f} | epoch={epoch} | path={save_path}")
        else:
            early_stop_counter += 1
            print(f"⏳ No improvement for {early_stop_counter} epoch(s)")

            if not overfit_debug and early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"🛑 Early stopping triggered (patience={EARLY_STOP_PATIENCE})")
                break

    print("\n🏁 Training Finished. Starting Final Test Evaluation...")

    if os.path.exists(save_path):
        print(f"📥 Loading Best Checkpoint: {save_path}")
        checkpoint = torch.load(save_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])

        test_loss, test_metrics = validate(
            model,
            test_loader,
            criterion,
            epoch="TEST",
            amp_enabled=amp_enabled,
            desc="[Test]",
        )

        print("\n" + "=" * 50)
        print(f"🏆 FINAL TAVAR FIXED RESULTS ON TEST SET (seed={seed})")
        print("=" * 50)
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Val SRCC: {best_srcc:.4f}")
        print(f"Test SRCC : {test_metrics['srcc']:.4f}")
        print(f"Test PCC  : {test_metrics['plcc']:.4f}")
        print(f"Test MAE  : {test_metrics['mae']:.4f}")
        print(f"Test MSE  : {test_metrics['mse']:.4f}")
        print("=" * 50)
    else:
        print("❌ Error: Checkpoint not found!")
        test_metrics = {"srcc": 0.0, "plcc": 0.0, "mae": 0.0, "mse": 0.0}
        test_loss = 0.0

    return (
        test_metrics["srcc"],
        test_metrics["plcc"],
        test_metrics["mae"],
        test_metrics["mse"],
        test_loss,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAVAR Baseline Training - Fixed Version")
    parser.add_argument("--seed", type=int, default=42, choices=SUPPORTED_SEEDS)
    parser.add_argument("--all_seeds", action="store_true", help="Run all seeds: 42, 123, 2024")
    parser.add_argument("--overfit_debug", action="store_true", help="Use first N training samples for overfit sanity check")
    parser.add_argument("--overfit_samples", type=int, default=256)
    args = parser.parse_args()

    run_seeds = SUPPORTED_SEEDS if args.all_seeds else [args.seed]

    results = []
    for seed in run_seeds:
        srcc, plcc, mae, mse, loss = main(
            seed=seed,
            overfit_debug=args.overfit_debug,
            overfit_samples=args.overfit_samples,
        )
        results.append({
            "seed": seed,
            "srcc": srcc,
            "plcc": plcc,
            "mae": mae,
            "mse": mse,
            "loss": loss,
        })

    if len(results) > 1:
        srcc_mean = np.mean([r["srcc"] for r in results])
        srcc_std = np.std([r["srcc"] for r in results])
        plcc_mean = np.mean([r["plcc"] for r in results])
        plcc_std = np.std([r["plcc"] for r in results])
        mae_mean = np.mean([r["mae"] for r in results])
        mae_std = np.std([r["mae"] for r in results])
        mse_mean = np.mean([r["mse"] for r in results])
        mse_std = np.std([r["mse"] for r in results])

        print("\n" + "=" * 60)
        print("📊 TAVAR Fixed Baseline Results Summary")
        print("=" * 60)
        print(f"Seeds: {run_seeds}")
        print("-" * 60)
        print(f"SRCC : {srcc_mean:.4f} ± {srcc_std:.4f}")
        print(f"PCC  : {plcc_mean:.4f} ± {plcc_std:.4f}")
        print(f"MAE  : {mae_mean:.4f} ± {mae_std:.4f}")
        print(f"MSE  : {mse_mean:.4f} ± {mse_std:.4f}")
        print("=" * 60)
