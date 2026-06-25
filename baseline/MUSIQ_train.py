# /root/autodl-tmp/pythonProject2/baseline/MUSIQ/MUSIQ_train.py
# ============================================================
# MUSIQ-style Multi-scale Baseline - Full-model Dataloader Version
#
# 修改目的：
# 1. 直接复用 train_full_model/full_model 使用的 get_feature_dataloader。
# 2. 不再自建 CSV Dataset，避免 split / label / transform 与 full_model 不一致。
# 3. 只使用 batch["image"]、batch["label"]、batch["label_raw"]，不使用 SSG-CAF 的
#    handcrafted / segmentation / region features。
# 4. 在 batch["image"] 基础上通过 interpolate 生成多尺度输入，作为 MUSIQ-style baseline。
# 5. 输出 normalized score，默认经过 sigmoid，降低预测崩到常数或异常尺度的风险。
# 6. 增加 pred_std 诊断；若 pred_std 极小，SRCC=0 是正常的数学结果。
# ============================================================

from __future__ import annotations

import os
import sys
import csv
import math
import time
import random
import argparse
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.amp import autocast, GradScaler


# ============================================================
# 0. 环境与路径
# ============================================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TORCH_WEIGHTS_ONLY"] = "0"

PROJECT_ROOT = "/root/autodl-tmp/pythonProject2"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    os.chdir(PROJECT_ROOT)
except Exception:
    pass

try:
    import utils.config as cfg
    from utils.data_loader import get_feature_dataloader
    print("✅ [Import] config and get_feature_dataloader loaded.")
except Exception:
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# 1. 参数
# ============================================================
SUPPORTED_SEEDS = [42, 123, 2026]

_raw_device = getattr(cfg, "DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
DEVICE = _raw_device if isinstance(_raw_device, torch.device) else torch.device(str(_raw_device))

CHECKPOINT_DIR = getattr(cfg, "CHECKPOINT_DIR", os.path.join(PROJECT_ROOT, "checkpoints/baseline"))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TRAIN_SPLIT = getattr(cfg, "TRAIN_SPLIT", "train")
VAL_SPLIT = getattr(cfg, "VAL_SPLIT", "val")
TEST_SPLIT = getattr(cfg, "TEST_SPLIT", "test")

DEFAULT_EPOCHS = int(getattr(cfg, "EPOCHS", 30))
DEFAULT_BATCH_SIZE = int(getattr(cfg, "BATCH_SIZE", 8))
DEFAULT_ACCUMULATION_STEPS = int(getattr(cfg, "ACCUMULATION_STEPS", 4))
DEFAULT_NUM_WORKERS = int(getattr(cfg, "NUM_WORKERS", 4))
DEFAULT_PIN_MEMORY = bool(getattr(cfg, "PIN_MEMORY", True))

GLOBAL_LOSS_TYPE = str(getattr(cfg, "GLOBAL_LOSS_TYPE", "smooth_l1")).lower()
GLOBAL_LOSS_BETA = float(getattr(cfg, "GLOBAL_LOSS_BETA", 0.05))
BEST_METRIC_SRCC_WEIGHT = float(getattr(cfg, "BEST_METRIC_SRCC_WEIGHT", 0.70))
EARLY_STOP_PATIENCE = int(getattr(cfg, "EARLY_STOP_PATIENCE", 10))
GRADIENT_CLIP = float(getattr(cfg, "GRADIENT_CLIP", 5.0))

USE_AMP = bool(getattr(getattr(cfg, "TrainConfig", object), "use_amp", True))
RESNET50_CKPT = getattr(cfg, "RESNET50_CKPT", None)

# MUSIQ-style transformer
N_LAYER = 6       # 原 MUSIQ 14 层很重，这里作为 baseline 用 6 层更稳、更省显存
D_HIDN = 384
N_HEAD = 6
D_FF = 768
D_MLP_HEAD = 512
GRID = 10
DROPOUT = 0.1
LN_EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description="MUSIQ-style baseline using full_model dataloader.")
    parser.add_argument("--seed", type=int, default=123, choices=SUPPORTED_SEEDS)
    parser.add_argument("--all_seeds", action="store_true")

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--accumulation_steps", type=int, default=DEFAULT_ACCUMULATION_STEPS)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)

    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_transformer", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=1)

    # 从 full_model 的 batch["image"] 生成多尺度。batch["image"] 通常已经是 224。
    parser.add_argument("--scale_main", type=int, default=224)
    parser.add_argument("--scale_1", type=int, default=192)
    parser.add_argument("--scale_2", type=int, default=160)

    parser.add_argument("--resnet_ckpt", type=str, default=RESNET50_CKPT)
    parser.add_argument("--save_prefix", type=str, default="musiq_style_full_loader_fair_stable")

    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_sigmoid", action="store_true", help="Disable sigmoid output. Not recommended for normalized labels.")
    parser.add_argument("--min_save_epoch", type=int, default=1, help="Earliest epoch allowed for model selection. Default 1 keeps the original fair protocol.")

    return parser.parse_args()


ARGS = parse_args()


def warn_if_debug_subset_used():
    """Warn when a run uses partial train/val/test batches.

    For paper results, max_train_batches, max_val_batches and max_test_batches
    must be None. Otherwise the result is only a debugging run and should not
    be reported in the baseline comparison table.
    """
    if ARGS.max_train_batches is not None or ARGS.max_val_batches is not None or ARGS.max_test_batches is not None:
        print("⚠️ Debug subset is enabled:")
        print(f"   max_train_batches={ARGS.max_train_batches}")
        print(f"   max_val_batches={ARGS.max_val_batches}")
        print(f"   max_test_batches={ARGS.max_test_batches}")
        print("   Do NOT report this run in the paper table.")



# ============================================================
# 2. 随机种子
# ============================================================
def set_global_seed(seed: int):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# 3. Dataloader：直接复用 full_model
# ============================================================
def build_dataloaders(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)

    common_kwargs = dict(
        batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=DEFAULT_PIN_MEMORY,
        worker_init_fn=seed_worker if ARGS.num_workers > 0 else None,
        run_tag="default",
        strict_scaler=True,
    )

    train_loader = get_feature_dataloader(
        split=TRAIN_SPLIT,
        shuffle=True,
        generator=g,
        **common_kwargs,
    )
    val_loader = get_feature_dataloader(
        split=VAL_SPLIT,
        shuffle=False,
        generator=g,
        **common_kwargs,
    )
    test_loader = get_feature_dataloader(
        split=TEST_SPLIT,
        shuffle=False,
        generator=g,
        **common_kwargs,
    )

    return train_loader, val_loader, test_loader


def inspect_first_batch(loader, tag: str):
    batch = next(iter(loader))
    print(f"\n🔍 [{tag} Batch Check]")
    print(f"   keys: {list(batch.keys())}")

    x = batch["image"]
    y = batch["label"].float().view(-1)

    print(f"   image: shape={tuple(x.shape)}, dtype={x.dtype}, min={float(x.min()):.4f}, max={float(x.max()):.4f}")
    print(f"   label_norm: min={float(y.min()):.4f}, max={float(y.max()):.4f}, mean={float(y.mean()):.4f}, std={float(y.std()):.4f}")

    if "label_raw" in batch:
        yr = batch["label_raw"].float().view(-1)
        print(f"   label_raw : min={float(yr.min()):.4f}, max={float(yr.max()):.4f}, mean={float(yr.mean()):.4f}, std={float(yr.std()):.4f}")


# ============================================================
# 4. Metrics
# ============================================================
def compute_metrics(
    preds_np: np.ndarray,
    targets_np: np.ndarray,
    acc_tols: Tuple[float, float] = (0.05, 0.10),
    acc_names: Tuple[str, str] = ("acc@0.05", "acc@0.10"),
) -> Dict[str, float]:
    preds = np.asarray(preds_np, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets_np, dtype=np.float64).reshape(-1)

    pred_std = float(np.std(preds))
    target_std = float(np.std(targets))

    if preds.shape[0] < 2:
        metrics = {"pearson": 0.0, "spearman": 0.0, "mse": 0.0, "mae": 0.0}
        for _, name in zip(acc_tols, acc_names):
            metrics[name] = 0.0
        metrics.update({"pred_std": pred_std, "target_std": target_std})
        return metrics

    if pred_std < 1e-12 or target_std < 1e-12:
        pcc = 0.0
        srcc = 0.0
    else:
        pcc = float(np.corrcoef(preds, targets)[0, 1])
        if np.isnan(pcc):
            pcc = 0.0

        srcc = float(spearmanr(preds, targets).correlation)
        if np.isnan(srcc):
            srcc = 0.0

    diff = preds - targets
    abs_diff = np.abs(diff)

    metrics = {
        "pearson": pcc,
        "spearman": srcc,
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(abs_diff)),
        "pred_std": pred_std,
        "target_std": target_std,
    }

    for tol, name in zip(acc_tols, acc_names):
        metrics[name] = float(np.mean(abs_diff <= tol))

    return metrics


def compute_selection_score(metrics: Dict[str, float]) -> float:
    return BEST_METRIC_SRCC_WEIGHT * float(metrics["spearman"]) + (1.0 - BEST_METRIC_SRCC_WEIGHT) * float(metrics["pearson"])


# ============================================================
# 5. Model
# ============================================================
def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def safe_load_resnet50(backbone: nn.Module, ckpt_path: Optional[str]) -> bool:
    if not ckpt_path:
        print("⚠️ [ResNet50] no local checkpoint path.")
        return False

    if not os.path.isfile(ckpt_path):
        print(f"⚠️ [ResNet50] checkpoint not found: {ckpt_path}")
        return False

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = extract_state_dict(ckpt)
    if not isinstance(state, dict):
        print(f"⚠️ [ResNet50] invalid checkpoint format: {ckpt_path}")
        return False

    model_state = backbone.state_dict()
    cleaned = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue

        key = k
        for prefix in ("module.", "model.", "backbone.", "features."):
            if key.startswith(prefix):
                key = key[len(prefix):]

        if key in model_state and model_state[key].shape == v.shape:
            cleaned[key] = v

    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    print(f"✅ [ResNet50] loaded local checkpoint: {ckpt_path}")
    print(f"   loaded shape-matched keys: {len(cleaned)} / {len(model_state)}")
    print(f"   missing={len(missing)}, unexpected={len(unexpected)}")
    return len(cleaned) > 0


class ResNet50Backbone(nn.Module):
    def __init__(self, ckpt_path: Optional[str] = None):
        super().__init__()

        net = resnet50(weights=None)
        loaded = safe_load_resnet50(net, ckpt_path)

        if not loaded:
            try:
                print("📥 [ResNet50] trying torchvision ImageNet weights...")
                net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                print("✅ [ResNet50] torchvision ImageNet weights loaded.")
                loaded = True
            except Exception as e:
                print(f"⚠️ [ResNet50] torchvision pretrained weights failed: {e}")
                print("   使用随机初始化 ResNet50，这会削弱 baseline；建议配置 cfg.RESNET50_CKPT。")

        self.pretrained_loaded = loaded
        self.features = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )

    def forward(self, x):
        return self.features(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_hidn: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_hidn % n_head == 0
        self.n_head = n_head
        self.d_head = d_hidn // n_head
        self.d_hidn = d_hidn

        self.W_Q = nn.Linear(d_hidn, d_hidn)
        self.W_K = nn.Linear(d_hidn, d_hidn)
        self.W_V = nn.Linear(d_hidn, d_hidn)
        self.linear = nn.Linear(d_hidn, d_hidn)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, q, k, v):
        B = q.size(0)
        q = self.W_Q(q).view(B, -1, self.n_head, self.d_head).transpose(1, 2)
        k = self.W_K(k).view(B, -1, self.n_head, self.d_head).transpose(1, 2)
        v = self.W_V(v).view(B, -1, self.n_head, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        ctx = torch.matmul(self.dropout(attn), v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, -1, self.d_hidn)
        return self.dropout(self.linear(ctx))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_hidn: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_hidn, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_hidn, eps=LN_EPS)
        self.ffn = nn.Sequential(
            nn.Linear(d_hidn, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_hidn),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_hidn, eps=LN_EPS)

    def forward(self, x):
        x = self.norm1(x + self.self_attn(x, x, x))
        x = self.norm2(x + self.ffn(x))
        return x


class MUSIQTransformer(nn.Module):
    def __init__(self, use_sigmoid: bool = True):
        super().__init__()

        self.use_sigmoid = bool(use_sigmoid)
        self.conv_reduce = nn.Conv2d(2048, D_HIDN, kernel_size=1, bias=False)

        self.scale_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(1, D_HIDN, 1, 1) * 0.02)
            for _ in range(3)
        ])
        self.spatial_emb = nn.Parameter(torch.randn(1, GRID, GRID, D_HIDN) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, D_HIDN) * 0.02)
        self.dropout = nn.Dropout(DROPOUT)

        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(D_HIDN, N_HEAD, D_FF, DROPOUT)
            for _ in range(N_LAYER)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(D_HIDN, eps=LN_EPS),
            nn.Linear(D_HIDN, D_MLP_HEAD),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MLP_HEAD, 1),
        )

        self._init_head()

    def _init_head(self):
        last = self.head[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=0.01)
            nn.init.constant_(last.bias, 0.0 if self.use_sigmoid else 0.5)

    def _add_spatial_emb(self, x):
        B, C, H, W = x.shape

        grid_h = torch.clamp(
            (torch.arange(H, device=x.device).float() / max(H, 1)) * GRID,
            0,
            GRID - 1,
        ).long()
        grid_w = torch.clamp(
            (torch.arange(W, device=x.device).float() / max(W, 1)) * GRID,
            0,
            GRID - 1,
        ).long()

        idx_h = grid_h.unsqueeze(1).expand(H, W)
        idx_w = grid_w.unsqueeze(0).expand(H, W)
        flat_idx = (idx_h * GRID + idx_w).view(1, H * W, 1).expand(B, H * W, C)

        flat_emb = self.spatial_emb.view(GRID * GRID, C)
        picked = flat_emb.unsqueeze(0).expand(B, -1, -1).gather(1, flat_idx)
        picked = picked.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return x + picked

    @staticmethod
    def _flatten(x):
        B, C, H, W = x.shape
        return x.view(B, C, H * W).permute(0, 2, 1).contiguous()

    def forward(self, feats: List[torch.Tensor]):
        B = feats[0].size(0)
        tokens = []

        for i, feat in enumerate(feats):
            feat = self.conv_reduce(feat)
            feat = feat + self.scale_embeds[i]
            feat = self._add_spatial_emb(feat)
            tokens.append(self._flatten(feat))

        x = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x)

        for layer in self.encoder:
            x = layer(x)

        logits = self.head(x[:, 0, :]).view(-1)

        if self.use_sigmoid:
            return torch.sigmoid(logits)

        return logits


class MUSIQStyleBaseline(nn.Module):
    def __init__(self, resnet_ckpt: Optional[str] = None, use_sigmoid: bool = True):
        super().__init__()
        self.backbone = ResNet50Backbone(ckpt_path=resnet_ckpt)
        self.transformer = MUSIQTransformer(use_sigmoid=use_sigmoid)

    @staticmethod
    def _resize(x: torch.Tensor, size: int):
        if x.shape[-1] == size and x.shape[-2] == size:
            return x
        return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

    def forward(self, image):
        img0 = self._resize(image, ARGS.scale_main)
        img1 = self._resize(image, ARGS.scale_1)
        img2 = self._resize(image, ARGS.scale_2)

        feats = [
            self.backbone(img0),
            self.backbone(img1),
            self.backbone(img2),
        ]

        return self.transformer(feats)


def freeze_backbone(model: MUSIQStyleBaseline):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.transformer.parameters():
        p.requires_grad = True

    frozen = sum(p.numel() for p in model.backbone.parameters())
    trainable = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
    print(f"🧊 Freeze backbone: frozen={frozen:,}, transformer trainable={trainable:,}")


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True
    print("🔥 Unfreeze all parameters.")


def make_optimizer(model: MUSIQStyleBaseline, lr_backbone: float, lr_transformer: float, weight_decay: float):
    backbone_decay, backbone_no_decay = [], []
    trans_decay, trans_no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        no_decay = (
            p.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "bn" in name.lower()
            or "emb" in name.lower()
            or "token" in name.lower()
        )

        if name.startswith("backbone."):
            if no_decay:
                backbone_no_decay.append(p)
            else:
                backbone_decay.append(p)
        else:
            if no_decay:
                trans_no_decay.append(p)
            else:
                trans_decay.append(p)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if trans_decay:
        groups.append({"params": trans_decay, "lr": lr_transformer, "weight_decay": weight_decay})
    if trans_no_decay:
        groups.append({"params": trans_no_decay, "lr": lr_transformer, "weight_decay": 0.0})

    if not groups:
        raise RuntimeError("No trainable parameters found.")

    optimizer = optim.AdamW(groups)

    print("\n🧩 Optimizer groups:")
    for i, g in enumerate(optimizer.param_groups):
        n_params = sum(p.numel() for p in g["params"])
        print(f"   group {i}: params={n_params:,}, lr={g['lr']:.2e}, wd={g['weight_decay']:.2e}")

    return optimizer


# ============================================================
# 6. Loss / Train / Eval
# ============================================================
def regression_loss(pred: torch.Tensor, target: torch.Tensor):
    if GLOBAL_LOSS_TYPE == "mse":
        return F.mse_loss(pred, target)
    return F.smooth_l1_loss(pred, target, beta=GLOBAL_LOSS_BETA)


def _forward_batch(model, batch, amp_enabled: bool):
    image = batch["image"].to(DEVICE, non_blocking=True).float()
    target_norm = batch["label"].to(DEVICE, non_blocking=True).float().view(-1)

    if "label_raw" in batch:
        target_raw = batch["label_raw"].to(DEVICE, non_blocking=True).float().view(-1)
    else:
        target_raw = target_norm * 9.0 + 1.0

    with autocast(device_type=DEVICE.type, enabled=amp_enabled):
        pred_norm = model(image).view(-1)
        loss = regression_loss(pred_norm, target_norm)

    pred_raw = pred_norm.detach().float() * 9.0 + 1.0
    return pred_norm, pred_raw, target_norm, target_raw, loss


def train_one_epoch(model, loader, optimizer, scaler, epoch: int, accumulation_steps: int, amp_enabled: bool, max_batches: Optional[int] = None):
    model.train()
    # 与常规 fine-tuning 一样，BN 统计保持稳定
    model.backbone.eval()

    total_loss = 0.0
    valid_batches = 0
    all_pred_norm, all_target_norm = [], []
    all_pred_raw, all_target_raw = [], []

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Ep {epoch} [Train]", leave=False, ncols=120)

    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        pred_norm, pred_raw, target_norm, target_raw, loss = _forward_batch(model, batch, amp_enabled)

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ NaN/Inf loss detected, skip batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss / accumulation_steps).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        valid_batches += 1

        all_pred_norm.append(pred_norm.detach().float().cpu().numpy())
        all_target_norm.append(target_norm.detach().float().cpu().numpy())
        all_pred_raw.append(pred_raw.cpu().numpy())
        all_target_raw.append(target_raw.detach().float().cpu().numpy())

        if i % 10 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", pred_std=f"{float(pred_norm.detach().std()):.4e}")

    if valid_batches == 0:
        metrics = {"spearman": 0.0, "pearson": 0.0, "mse": 0.0, "mae": 0.0, "pred_std": 0.0, "target_std": 0.0}
        return metrics, 0.0

    pred_norm_np = np.concatenate(all_pred_norm)
    target_norm_np = np.concatenate(all_target_norm)
    pred_raw_np = np.concatenate(all_pred_raw)
    target_raw_np = np.concatenate(all_target_raw)

    metrics = compute_metrics(pred_norm_np, target_norm_np)
    raw_metrics = compute_metrics(pred_raw_np, target_raw_np, acc_tols=(0.5, 1.0), acc_names=("acc@0.5", "acc@1.0"))

    metrics["mse"] = raw_metrics["mse"]
    metrics["mae"] = raw_metrics["mae"]
    metrics["acc@0.5"] = raw_metrics["acc@0.5"]
    metrics["acc@1.0"] = raw_metrics["acc@1.0"]

    avg_loss = total_loss / valid_batches

    print(
        f"📊 [Train] Ep {epoch}: Loss={avg_loss:.4f} | "
        f"SRCC={metrics['spearman']:.4f} | PCC={metrics['pearson']:.4f} | "
        f"MSE={metrics['mse']:.4f} | MAE={metrics['mae']:.4f} | "
        f"pred_norm_mean={pred_norm_np.mean():.4f}, pred_norm_std={pred_norm_np.std():.6f}, "
        f"target_norm_mean={target_norm_np.mean():.4f}, target_norm_std={target_norm_np.std():.6f}"
    )

    if metrics["pred_std"] < 1e-6:
        print("⚠️ Train prediction std is near zero; SRCC will be 0. Check learning rate, frozen backbone, and output head.")

    return metrics, avg_loss


@torch.no_grad()
def evaluate(model, loader, epoch, amp_enabled: bool, desc: str, max_batches: Optional[int] = None, save_path: Optional[str] = None):
    model.eval()

    total_loss = 0.0
    valid_batches = 0
    all_pred_norm, all_target_norm = [], []
    all_pred_raw, all_target_raw = [], []
    all_image_ids = []

    pbar = tqdm(loader, desc=f"Ep {epoch} {desc}", leave=False, ncols=120)

    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        pred_norm, pred_raw, target_norm, target_raw, loss = _forward_batch(model, batch, amp_enabled)

        total_loss += float(loss.item())
        valid_batches += 1

        all_pred_norm.append(pred_norm.detach().float().cpu().numpy())
        all_target_norm.append(target_norm.detach().float().cpu().numpy())
        all_pred_raw.append(pred_raw.cpu().numpy())
        all_target_raw.append(target_raw.detach().float().cpu().numpy())

        if "image_id" in batch:
            ids = batch["image_id"]
            if isinstance(ids, list):
                all_image_ids.extend([str(x) for x in ids])
            else:
                all_image_ids.extend([str(x.item()) if hasattr(x, "item") else str(x) for x in ids])
        else:
            bs = pred_norm.shape[0]
            start = sum(len(a) for a in all_pred_norm[:i])
            all_image_ids.extend([f"idx_{start + j}" for j in range(bs)])

    if valid_batches == 0:
        metrics = {"spearman": 0.0, "pearson": 0.0, "mse": 0.0, "mae": 0.0, "pred_std": 0.0, "target_std": 0.0}
        return metrics, 0.0

    pred_norm_np = np.concatenate(all_pred_norm)
    target_norm_np = np.concatenate(all_target_norm)
    pred_raw_np = np.concatenate(all_pred_raw)
    target_raw_np = np.concatenate(all_target_raw)

    metrics = compute_metrics(pred_norm_np, target_norm_np)
    raw_metrics = compute_metrics(pred_raw_np, target_raw_np, acc_tols=(0.5, 1.0), acc_names=("acc@0.5", "acc@1.0"))

    metrics["mse"] = raw_metrics["mse"]
    metrics["mae"] = raw_metrics["mae"]
    metrics["acc@0.5"] = raw_metrics["acc@0.5"]
    metrics["acc@1.0"] = raw_metrics["acc@1.0"]

    avg_loss = total_loss / valid_batches

    print(
        f"✨ {desc} Ep {epoch}: Loss={avg_loss:.4f} | "
        f"SRCC={metrics['spearman']:.4f} | PCC={metrics['pearson']:.4f} | "
        f"MSE={metrics['mse']:.4f} | MAE={metrics['mae']:.4f} | "
        f"pred_norm_mean={pred_norm_np.mean():.4f}, pred_norm_std={pred_norm_np.std():.6f}, "
        f"target_norm_mean={target_norm_np.mean():.4f}, target_norm_std={target_norm_np.std():.6f}"
    )

    if metrics["pred_std"] < 1e-6:
        print(f"⚠️ {desc} prediction std is near zero; SRCC will be 0.")

    if save_path and len(all_image_ids) == len(pred_raw_np):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame({
            "image_id": all_image_ids,
            "label_raw": target_raw_np,
            "pred_raw": pred_raw_np,
        })
        df.to_csv(save_path, index=False)
        print(f"💾 Per-sample predictions saved: {save_path} ({len(df)} rows)")

    return metrics, avg_loss


def save_history_csv(history_rows: List[Dict], path: str):
    if not history_rows:
        return

    fieldnames = sorted({k for row in history_rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)


# ============================================================
# 7. Main
# ============================================================
def run_one_seed(seed: int):
    print("\n" + "=" * 80)
    print(f"🎲 MUSIQ-style Baseline Full-loader Fixed | seed={seed}")
    print("=" * 80)

    set_global_seed(seed)
    amp_enabled = bool(USE_AMP and (not ARGS.no_amp) and DEVICE.type == "cuda")

    print(f"🖥️ Device: {DEVICE}")
    print(f"📥 RESNET50_CKPT={ARGS.resnet_ckpt}")
    print(f"🖼️ scales from batch['image']: {ARGS.scale_main}, {ARGS.scale_1}, {ARGS.scale_2}")
    print(f"🔒 sigmoid_output={not ARGS.no_sigmoid}")

    train_loader, val_loader, test_loader = build_dataloaders(seed)
    inspect_first_batch(train_loader, "Train")
    inspect_first_batch(val_loader, "Val")

    print("\n🏗️ Building MUSIQ-style baseline...")
    model = MUSIQStyleBaseline(
        resnet_ckpt=ARGS.resnet_ckpt,
        use_sigmoid=(not ARGS.no_sigmoid),
    ).to(DEVICE)

    if ARGS.freeze_backbone_epochs > 0 and getattr(model.backbone, "pretrained_loaded", False):
        freeze_backbone(model)
    else:
        if ARGS.freeze_backbone_epochs > 0:
            print("⚠️ ResNet50 pretrained weights were not loaded, so backbone will not be frozen.")
        unfreeze_all(model)

    optimizer = make_optimizer(
        model=model,
        lr_backbone=ARGS.lr_backbone,
        lr_transformer=ARGS.lr_transformer,
        weight_decay=ARGS.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        threshold=1e-4,
        min_lr=1e-7,
    )

    scaler = GradScaler(DEVICE.type, enabled=amp_enabled)

    save_path = os.path.join(CHECKPOINT_DIR, f"{ARGS.save_prefix}_seed{seed}.pth")
    history_csv = os.path.join(CHECKPOINT_DIR, f"{ARGS.save_prefix}_seed{seed}_history.csv")

    best_selection = -1.0
    best_metrics = None
    best_epoch = -1
    early_stop = 0
    history_rows = []

    print(
        f"\n🚀 Start Training: epochs={ARGS.epochs}, batch={ARGS.batch_size}, accum={ARGS.accumulation_steps}, "
        f"lr_backbone={ARGS.lr_backbone}, lr_transformer={ARGS.lr_transformer}, wd={ARGS.weight_decay}, "
        f"freeze_backbone_epochs={ARGS.freeze_backbone_epochs}, min_save_epoch={ARGS.min_save_epoch}, AMP={amp_enabled}"
    )

    for epoch in range(1, ARGS.epochs + 1):
        if ARGS.freeze_backbone_epochs > 0 and epoch == ARGS.freeze_backbone_epochs + 1:
            unfreeze_all(model)
            optimizer = make_optimizer(
                model=model,
                lr_backbone=ARGS.lr_backbone,
                lr_transformer=ARGS.lr_transformer,
                weight_decay=ARGS.weight_decay,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=1,
                threshold=1e-4,
                min_lr=1e-7,
            )

        train_metrics, train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            accumulation_steps=ARGS.accumulation_steps,
            amp_enabled=amp_enabled,
            max_batches=ARGS.max_train_batches,
        )

        val_metrics, val_loss = evaluate(
            model=model,
            loader=val_loader,
            epoch=epoch,
            amp_enabled=amp_enabled,
            desc="[Val]",
            max_batches=ARGS.max_val_batches,
        )

        selection = compute_selection_score(val_metrics)
        scheduler.step(selection)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_srcc": train_metrics["spearman"],
            "train_pcc": train_metrics["pearson"],
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "train_pred_std": train_metrics["pred_std"],
            "val_srcc": val_metrics["spearman"],
            "val_pcc": val_metrics["pearson"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_pred_std": val_metrics["pred_std"],
            "val_selection": selection,
            "lr_min": min(pg["lr"] for pg in optimizer.param_groups),
            "lr_max": max(pg["lr"] for pg in optimizer.param_groups),
        }
        history_rows.append(row)
        save_history_csv(history_rows, history_csv)

        print(
            f"🎯 [Val Selection] Ep {epoch}: Sel={selection:.4f} | Best={best_selection:.4f} | "
            f"SRCC={val_metrics['spearman']:.4f} | PCC={val_metrics['pearson']:.4f} | "
            f"pred_std={val_metrics['pred_std']:.6f}"
        )

        can_select_this_epoch = epoch >= int(ARGS.min_save_epoch)

        if (selection > best_selection + 1e-4) and can_select_this_epoch:
            best_selection = selection
            best_metrics = val_metrics
            best_epoch = epoch
            early_stop = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "val_loss": val_loss,
                    "val_selection": selection,
                    "epoch": epoch,
                    "seed": seed,
                    "args": vars(ARGS),
                    "history_csv": history_csv,
                },
                save_path,
            )
            print(f"🏆 New Best Saved: {save_path}")
        elif not can_select_this_epoch:
            print(f"⏸️ Epoch {epoch} is before min_save_epoch={ARGS.min_save_epoch}; not eligible for model selection.")
        else:
            early_stop += 1
            print(f"⏳ No improvement: {early_stop}/{EARLY_STOP_PATIENCE}")

        if early_stop >= EARLY_STOP_PATIENCE:
            print("🛑 Early stopping.")
            break

    print("\n🏁 Training finished. Loading best checkpoint for test...")
    if not os.path.exists(save_path):
        print("❌ No checkpoint saved.")
        return {
            "seed": seed,
            "best_epoch": best_epoch,
            "val_selection": best_selection,
            "pearson": 0.0,
            "spearman": 0.0,
            "mse": 0.0,
            "mae": 0.0,
        }

    ckpt = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    test_pred_csv = os.path.join(CHECKPOINT_DIR, "predictions", f"MUSIQ_style_seed{seed}_test_predictions.csv")
    os.makedirs(os.path.dirname(test_pred_csv), exist_ok=True)

    test_metrics, test_loss = evaluate(
        model=model,
        loader=test_loader,
        epoch="TEST",
        amp_enabled=amp_enabled,
        desc="[Test]",
        max_batches=ARGS.max_test_batches,
        save_path=test_pred_csv,
    )

    print("\n" + "=" * 70)
    print(f"🏆 FINAL MUSIQ-STYLE FULL-LOADER BASELINE TEST RESULTS | seed={seed}")
    print("=" * 70)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Selection: {best_selection:.4f}")
    print(f"Best Val SRCC: {best_metrics['spearman']:.4f}" if best_metrics else "Best Val SRCC: N/A")
    print(f"Test SRCC : {test_metrics['spearman']:.4f}")
    print(f"Test PCC  : {test_metrics['pearson']:.4f}")
    print(f"Test MSE  : {test_metrics['mse']:.4f}")
    print(f"Test MAE  : {test_metrics['mae']:.4f}")
    print(f"Test pred_std: {test_metrics['pred_std']:.6f}")
    print("=" * 70)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_selection": best_selection,
        "val_srcc": best_metrics["spearman"] if best_metrics else 0.0,
        "val_pcc": best_metrics["pearson"] if best_metrics else 0.0,
        "pearson": test_metrics["pearson"],
        "spearman": test_metrics["spearman"],
        "mse": test_metrics["mse"],
        "mae": test_metrics["mae"],
    }


def print_summary(results: List[Dict]):
    if len(results) == 1:
        return

    print("\n" + "=" * 70)
    print("📊 MUSIQ-style Full-loader Multi-seed Summary")
    print("Protocol: same full-model dataloader, same split, image-only input, no semantic/region/handcrafted features.")
    print("=" * 70)

    for key in ["spearman", "pearson", "mse", "mae", "val_srcc", "val_pcc"]:
        values = np.array([r[key] for r in results], dtype=float)
        print(f"{key:10s}: {values.mean():.4f} ± {values.std(ddof=1):.4f}")

    print("=" * 70)


if __name__ == "__main__":
    seeds = SUPPORTED_SEEDS if ARGS.all_seeds else [ARGS.seed]
    results = []

    for seed in seeds:
        result = run_one_seed(seed)
        results.append(result)

    print_summary(results)
