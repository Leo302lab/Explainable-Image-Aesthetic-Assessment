
from __future__ import annotations

import os
import sys
import csv
import json
import random
import math
import traceback
import argparse
import importlib
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============================================================
# 环境设置
# ============================================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TORCH_WEIGHTS_ONLY"] = "0"


def super_setup():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        os.chdir(project_root)
    except Exception:
        pass


super_setup()

# 添加命令行参数解析
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="训练完整美学评估模型")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--save_name", type=str, default="full_aesthetic_model.pth", help="模型保存名称")
    return parser.parse_args()

ARGS = parse_args()

def _detect_config_module() -> str:
    if "--config_module" in sys.argv:
        idx = sys.argv.index("--config_module")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return os.environ.get("AESTHETIC_CONFIG_MODULE", "utils.config")

CONFIG_MODULE = _detect_config_module()

try:
    cfg = importlib.import_module(CONFIG_MODULE)
    ModelConfig = getattr(cfg, "ModelConfig")
    from utils.data_loader import get_feature_dataloader
    from models.full_model import FullAestheticModel

    print(f"✅ [Import] 核心模块导入成功 | cfg={CONFIG_MODULE}")
except Exception:
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# 0. 配置读取
# ============================================================
DEVICE = cfg.DEVICE
CHECKPOINT_DIR = cfg.CHECKPOINT_DIR

EPOCHS = int(cfg.EPOCHS)
BATCH_SIZE = int(cfg.BATCH_SIZE)
ACCUMULATION_STEPS = int(cfg.ACCUMULATION_STEPS)

LR = float(cfg.LR)
WEIGHT_DECAY = float(cfg.WEIGHT_DECAY)
GRADIENT_CLIP = float(cfg.GRADIENT_CLIP)

NUM_WORKERS = int(cfg.NUM_WORKERS)
PIN_MEMORY = bool(cfg.PIN_MEMORY)

TRAIN_SPLIT = cfg.TRAIN_SPLIT
VAL_SPLIT = cfg.VAL_SPLIT
TEST_SPLIT = cfg.TEST_SPLIT

EARLY_STOP_PATIENCE = int(cfg.EARLY_STOP_PATIENCE)
# 优先使用命令行参数传入的种子
SEED = int(ARGS.seed) if ARGS.seed is not None else int(cfg.SEED)

GLOBAL_FEAT_DIM = int(getattr(cfg, "GLOBAL_CORE_DIM", getattr(cfg, "GLOBAL_FEATURE_DIM", 7)))

WARMUP_EPOCHS = int(getattr(cfg, "WARMUP_EPOCHS", 2))
PHASE2_EPOCHS = int(getattr(cfg, "PHASE2_EPOCHS", 4))
PHASE3_EPOCHS = int(
    getattr(
        cfg,
        "PHASE3_EPOCHS",
        max(int(cfg.EPOCHS) - int(getattr(cfg, "WARMUP_EPOCHS", 2)) - int(getattr(cfg, "PHASE2_EPOCHS", 6)), 0),
    )
)

WARMUP_LR_SCALE = float(getattr(cfg, "WARMUP_LR_SCALE", 0.10))
SCHEDULER_TYPE_DEFAULT = str(getattr(cfg, "SCHEDULER_TYPE", "plateau")).lower()
MIN_LR = float(getattr(cfg, "MIN_LR", getattr(getattr(cfg, "TrainConfig", object), "min_lr", 1e-6)))
PLATEAU_FACTOR = float(getattr(cfg, "PLATEAU_FACTOR", 0.5))
PLATEAU_PATIENCE = int(getattr(cfg, "PLATEAU_PATIENCE", 1))
PLATEAU_THRESHOLD = float(getattr(cfg, "PLATEAU_THRESHOLD", 1e-4))

GLOBAL_LOSS_TYPE = str(getattr(cfg, "GLOBAL_LOSS_TYPE", "smooth_l1")).lower()
GLOBAL_LOSS_BETA = float(getattr(cfg, "GLOBAL_LOSS_BETA", 0.05))

LOSS_GLOBAL_WEIGHT = float(getattr(cfg, "LOSS_GLOBAL_WEIGHT", 1.0))
LOSS_CONSISTENCY_WEIGHT = float(getattr(cfg, "LOSS_CONSISTENCY_WEIGHT", 0.0))
LOSS_ALIGNMENT_WEIGHT = float(getattr(cfg, "LOSS_ALIGNMENT_WEIGHT", 0.0))
LOSS_REGION_QUALITY_WEIGHT = float(getattr(cfg, "LOSS_REGION_QUALITY_WEIGHT", 0.0))
LOSS_RANKING_WEIGHT = float(getattr(cfg, "LOSS_RANKING_WEIGHT", 0.02))

PHASE2_CONSISTENCY_START = float(getattr(cfg, "PHASE2_CONSISTENCY_START", max(0.0, LOSS_CONSISTENCY_WEIGHT * 0.5)))
PHASE2_CONSISTENCY_END = float(getattr(cfg, "PHASE2_CONSISTENCY_END", LOSS_CONSISTENCY_WEIGHT))
PHASE2_ALIGNMENT_START = float(getattr(cfg, "PHASE2_ALIGNMENT_START", max(0.0, LOSS_ALIGNMENT_WEIGHT * 0.5)))
PHASE2_ALIGNMENT_END = float(getattr(cfg, "PHASE2_ALIGNMENT_END", LOSS_ALIGNMENT_WEIGHT))
PHASE2_REGION_QUALITY_START = float(getattr(cfg, "PHASE2_REGION_QUALITY_START", max(0.0, LOSS_REGION_QUALITY_WEIGHT * 0.5)))
PHASE2_REGION_QUALITY_END = float(getattr(cfg, "PHASE2_REGION_QUALITY_END", LOSS_REGION_QUALITY_WEIGHT))
PHASE2_RANKING_START = float(getattr(cfg, "PHASE2_RANKING_START", 0.0))
PHASE2_RANKING_END = float(getattr(cfg, "PHASE2_RANKING_END", LOSS_RANKING_WEIGHT))

PHASE3_CONSISTENCY_START = float(getattr(cfg, "PHASE3_CONSISTENCY_START", PHASE2_CONSISTENCY_END))
PHASE3_CONSISTENCY_END = float(getattr(cfg, "PHASE3_CONSISTENCY_END", max(PHASE2_CONSISTENCY_END, LOSS_CONSISTENCY_WEIGHT)))
PHASE3_ALIGNMENT_START = float(getattr(cfg, "PHASE3_ALIGNMENT_START", PHASE2_ALIGNMENT_END))
PHASE3_ALIGNMENT_END = float(getattr(cfg, "PHASE3_ALIGNMENT_END", max(PHASE2_ALIGNMENT_END, LOSS_ALIGNMENT_WEIGHT)))
PHASE3_REGION_QUALITY_START = float(getattr(cfg, "PHASE3_REGION_QUALITY_START", PHASE2_REGION_QUALITY_END))
PHASE3_REGION_QUALITY_END = float(getattr(cfg, "PHASE3_REGION_QUALITY_END", max(PHASE2_REGION_QUALITY_END, LOSS_REGION_QUALITY_WEIGHT)))
PHASE3_RANKING_START = float(getattr(cfg, "PHASE3_RANKING_START", PHASE2_RANKING_END))
PHASE3_RANKING_END = float(getattr(cfg, "PHASE3_RANKING_END", max(PHASE2_RANKING_END, LOSS_RANKING_WEIGHT)))

RANKING_PAIR_THRESHOLD = float(
    getattr(
        cfg,
        "RANKING_PAIR_THRESHOLD",
        0.05 if bool(getattr(cfg, "LABEL_IS_NORMALIZED_0_1", True)) else 0.5,
    )
)
RANKING_MARGIN = float(
    getattr(
        cfg,
        "RANKING_MARGIN",
        0.03 if bool(getattr(cfg, "LABEL_IS_NORMALIZED_0_1", True)) else 0.3,
    )
)

BEST_METRIC_SRCC_WEIGHT = float(getattr(cfg, "BEST_METRIC_SRCC_WEIGHT", 0.70))

LABEL_IS_NORMALIZED_0_1 = bool(getattr(cfg, "LABEL_IS_NORMALIZED_0_1", True))
LABEL_RAW_SCALE = float(getattr(cfg, "LABEL_RAW_SCALE", 10.0))
CSV_SUB_FEATS = list(getattr(cfg, "CSV_SUB_FEATS", []))
SAVE_BEST_ONLY = bool(getattr(cfg, "SAVE_BEST_ONLY", True))

SEG_LR_SCALE = float(getattr(cfg, "SEG_LR_SCALE", 0.0))

PHASE1_HEAD_LR = float(getattr(cfg, "PHASE1_HEAD_LR", LR * WARMUP_LR_SCALE))
PHASE2_HEAD_LR = float(getattr(cfg, "PHASE2_HEAD_LR", LR))
PHASE2_VIT_LR = float(getattr(cfg, "PHASE2_VIT_LR", LR * float(getattr(cfg, "VIT_LR_SCALE", 0.1))))
PHASE2_RESNET_LR = float(getattr(cfg, "PHASE2_RESNET_LR", 0.0))
PHASE3_HEAD_LR = float(getattr(cfg, "PHASE3_HEAD_LR", LR * 0.125))
PHASE3_VIT_LR = float(getattr(cfg, "PHASE3_VIT_LR", max(PHASE2_VIT_LR * 0.25, MIN_LR)))
PHASE3_RESNET_LR = float(getattr(cfg, "PHASE3_RESNET_LR", max(LR * 0.01, MIN_LR)))

PHASE2_RESNET_UNFREEZE_MODE = str(getattr(cfg, "PHASE2_RESNET_UNFREEZE_MODE", "none")).lower()
RESNET_UNFREEZE_MODE = str(getattr(cfg, "RESNET_UNFREEZE_MODE", "layer4")).lower()
VIT_UNFREEZE_START_LAYER = int(getattr(cfg, "VIT_UNFREEZE_START_LAYER", 9))
SEG_FREEZE = bool(getattr(cfg, "SEG_FREEZE", True))
PHASE3_ONLY_IF_PHASE2_IMPROVED = bool(getattr(cfg, "PHASE3_ONLY_IF_PHASE2_IMPROVED", True))
FREEZE_BATCHNORM_STATS = bool(getattr(cfg, "FREEZE_BATCHNORM_STATS", True))

USE_OFFLINE_SEG_MASK = bool(getattr(cfg, "USE_OFFLINE_SEG_MASK", True))
USE_AMP = bool(getattr(getattr(cfg, "TrainConfig", object), "use_amp", True))

TRAIN_CFG = getattr(cfg, "TrainConfig", None)
BETA1 = float(getattr(TRAIN_CFG, "beta1", 0.9)) if TRAIN_CFG is not None else 0.9
BETA2 = float(getattr(TRAIN_CFG, "beta2", 0.999)) if TRAIN_CFG is not None else 0.999
EPS = float(getattr(TRAIN_CFG, "eps", 1e-8)) if TRAIN_CFG is not None else 1e-8

BACKBONE_WEIGHT_DECAY = float(getattr(cfg, "BACKBONE_WEIGHT_DECAY", WEIGHT_DECAY))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# 1. 模型构建
# ============================================================
def build_full_model(device, ablation_type="full"):
    model_cfg = ModelConfig()

    model_cfg.seg_model_name_or_path = cfg.SEGFORMER_BASE_DIR
    model_cfg.seg_finetuned_ckpt = cfg.SEGFORMER_FINETUNE_CKPT
    model_cfg.resnet50_ckpt = cfg.RESNET50_CKPT
    model_cfg.vit_ckpt_path = cfg.VIT_BASE_CKPT
    model_cfg.num_regions = cfg.SEG_NUM_CLASSES

    model_cfg.region_feature_dim = cfg.REGION_FEATURE_DIM
    model_cfg.global_feature_dim = cfg.FUSED_GLOBAL_DIM
    model_cfg.handcrafted_global_dim = cfg.GLOBAL_HANDCRAFTED_DIM
    model_cfg.handcrafted_region_dim = cfg.REGION_HANDCRAFTED_DIM

    model_cfg.region_type_embed_dim = int(getattr(cfg, "REGION_TYPE_EMBED_DIM", 0))
    model_cfg.global_feature_names = list(getattr(cfg, "GLOBAL_FEATURE_NAMES", []))
    model_cfg.region_feature_names = list(getattr(cfg, "REGION_FEATURE_NAMES", []))
    model_cfg.region_names = list(getattr(cfg, "REGION_NAMES", []))

    model_cfg.cross_attn_heads = cfg.ATTENTION_HEADS
    model_cfg.cross_attn_dropout = cfg.ATTENTION_DROPOUT
    model_cfg.region_dropout_prob = cfg.REGION_DROPOUT_PROB

    model_cfg.freeze_seg = cfg.SEG_FREEZE
    model_cfg.freeze_vit = cfg.FREEZE_VIT
    model_cfg.use_online_handcrafted_extractor = bool(getattr(cfg, "USE_ONLINE_HANDCRAFTED_EXTRACTOR", False))
    model_cfg.disable_regions = False

    model_cfg.use_roi_align_region_features = bool(getattr(cfg, "USE_ROI_ALIGN_REGION_FEATURES", True))
    model_cfg.roi_output_size = int(getattr(cfg, "ROI_OUTPUT_SIZE", 7))
    model_cfg.roi_sampling_ratio = int(getattr(cfg, "ROI_SAMPLING_RATIO", 2))
    model_cfg.roi_align_binarize_threshold = float(getattr(cfg, "ROI_ALIGN_BINARIZE_THRESHOLD", 0.5))
    model_cfg.roi_min_box_size = float(getattr(cfg, "ROI_MIN_BOX_SIZE", 1.0))

    expected_region_count = int(getattr(cfg, "REGION_COUNT", model_cfg.num_regions))
    if expected_region_count != model_cfg.num_regions:
        raise ValueError(
            f"REGION_COUNT ({expected_region_count}) and SEG_NUM_CLASSES ({model_cfg.num_regions}) must match."
        )

    if ablation_type in {"no_seg", "no_region_features"}:
        model_cfg.ablate_segmentation = True
        model_cfg.disable_regions = True
    elif ablation_type == "no_cross_att":
        model_cfg.ablate_cross_attention = True

    model = FullAestheticModel(cfg=model_cfg).to(device)
    return model

# ============================================================
# 2. System Check
# ============================================================
def _branch_status(module):
    if module is None:
        return "❌ Missing"

    total_params = sum(p.numel() for p in module.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6

    if total_params == 0:
        return "❌ Empty"

    mode = "Trainable" if trainable_params > 0 else "Frozen"
    return f"✅ Loaded | {mode} | {total_params:6.2f}M"


def print_system_check(model):
    print("\n🔍 [System Check] Branches...")
    print(f"ResNet     | {_branch_status(getattr(model, 'cnn_backbone', None))}")
    print(f"ViT        | {_branch_status(getattr(model, 'vit', None))}")
    print(f"Seg        | {_branch_status(getattr(model, 'seg_branch', None))}")

    print("\n🧩 Segmentation Path:")
    print(f"   USE_OFFLINE_SEG_MASK           = {cfg.USE_OFFLINE_SEG_MASK}")
    print(f"   SEG_FREEZE                     = {cfg.SEG_FREEZE}")
    print(f"   PHASE2_RESNET_MODE             = {PHASE2_RESNET_UNFREEZE_MODE}")
    print(f"   PHASE3_RESNET_MODE             = {RESNET_UNFREEZE_MODE}")
    print(f"   VIT_UNFREEZE_START_LAYER       = {VIT_UNFREEZE_START_LAYER}")
    print(f"   GLOBAL_LOSS_TYPE               = {GLOBAL_LOSS_TYPE}")
    print(f"   USE_ROI_ALIGN_REGION_FEATURES  = {getattr(cfg, 'USE_ROI_ALIGN_REGION_FEATURES', True)}")
    print(f"   REGION_PRESENCE_MIN_AREA_RATIO = {getattr(cfg, 'REGION_PRESENCE_MIN_AREA_RATIO', 0.002)}")
    print(f"   ROI_ADAPTIVE_THRESHOLD_RATIO   = {getattr(cfg, 'ROI_ADAPTIVE_THRESHOLD_RATIO', 0.5)}")
    print(f"   ATTENTION_DROPOUT              = {getattr(cfg, 'ATTENTION_DROPOUT', 0.2)}")
    print(f"   REGION_DROPOUT_PROB            = {getattr(cfg, 'REGION_DROPOUT_PROB', 0.3)}")

# ============================================================
# 3. 损失函数
# ============================================================
class DesignAlignedLoss(nn.Module):
    def __init__(
        self,
        global_loss_type: str = "smooth_l1",
        global_loss_beta: float = 0.05,
        global_weight: float = 1.0,
        consistency_weight: float = 0.05,
        alignment_weight: float = 0.02,
        region_quality_weight: float = 0.0,
        ranking_weight: float = 0.10,
        ranking_margin: float = 0.03,
        ranking_pair_threshold: float = 0.05,
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.global_loss_type = str(global_loss_type).lower()
        self.global_loss_beta = float(global_loss_beta)
        self.global_weight = float(global_weight)
        self.consistency_weight = float(consistency_weight)
        self.alignment_weight = float(alignment_weight)
        self.region_quality_weight = float(region_quality_weight)
        self.ranking_weight = float(ranking_weight)
        self.ranking_margin = float(ranking_margin)
        self.ranking_pair_threshold = float(ranking_pair_threshold)
        self.feature_names = [str(x).lower() for x in (feature_names or [])]
        self.name_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}

    @staticmethod
    def _reduce_attention_to_region_level(
        region_attention_weights: Optional[torch.Tensor],
        region_padding_mask: Optional[torch.Tensor],
        head_importance: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if region_attention_weights is None:
            return None

        if region_attention_weights.ndim == 2:
            region_w = region_attention_weights
        elif region_attention_weights.ndim == 3:
            if head_importance is not None:
                head_w = head_importance.unsqueeze(-1)
                region_w = (region_attention_weights * head_w).sum(dim=1)
            else:
                region_w = region_attention_weights.mean(dim=1)
        else:
            raise ValueError(f"region_attention_weights shape unsupported: {tuple(region_attention_weights.shape)}")

        if region_padding_mask is not None:
            region_w = region_w.masked_fill(region_padding_mask, 0.0)
        return region_w

    def _compute_region_weights(
        self,
        region_scores: torch.Tensor,
        region_attention_weights: Optional[torch.Tensor],
        region_padding_mask: Optional[torch.Tensor],
        head_importance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, num_regions = region_scores.shape
        device = region_scores.device

        if region_padding_mask is None:
            region_padding_mask = torch.zeros(bsz, num_regions, dtype=torch.bool, device=device)

        valid_mask = ~region_padding_mask

        region_attention_weights = self._reduce_attention_to_region_level(
            region_attention_weights=region_attention_weights,
            region_padding_mask=region_padding_mask,
            head_importance=head_importance,
        )

        if region_attention_weights is not None:
            w = region_attention_weights.clone().masked_fill(region_padding_mask, 0.0)
            w_sum = w.sum(dim=1, keepdim=True)
            has_valid_weight = w_sum > 1e-8
            w_norm = torch.where(has_valid_weight, w / (w_sum + 1e-8), torch.zeros_like(w))

            uniform_w = valid_mask.float()
            uniform_sum = uniform_w.sum(dim=1, keepdim=True).clamp_min(1.0)
            uniform_w = uniform_w / uniform_sum
            w_final = torch.where(has_valid_weight, w_norm, uniform_w)
        else:
            w_final = valid_mask.float()
            w_sum = w_final.sum(dim=1, keepdim=True).clamp_min(1.0)
            w_final = w_final / w_sum

        return w_final

    def _global_regression_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.global_loss_type == "mse":
            return F.mse_loss(preds, targets)
        if self.global_loss_type == "smooth_l1":
            return F.smooth_l1_loss(preds, targets, beta=self.global_loss_beta)
        raise ValueError(f"不支持的 GLOBAL_LOSS_TYPE: {self.global_loss_type}")

    def _get_feat(self, feats: torch.Tensor, names: tuple[str, ...]) -> Optional[torch.Tensor]:
        for name in names:
            if name in self.name_to_idx and self.name_to_idx[name] < feats.shape[-1]:
                return feats[..., self.name_to_idx[name]]
        return None

    @staticmethod
    def _to_unit_interval(x: torch.Tensor, feat_type: str) -> torch.Tensor:
        x = x.clone()
        if feat_type in {"laplacian_norm", "sharpness_norm", "edge_sharpness", "nr_sharpness"}:
            return (x / 10.0).clamp(0.0, 1.0)
        return x.clamp(0.0, 1.0)

    def _build_region_quality_target(
        self,
        hand_region_raw: Optional[torch.Tensor],
        region_padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if hand_region_raw is None or hand_region_raw.ndim != 3:
            return None

        feats = hand_region_raw
        device = feats.device
        bsz, num_regions, _ = feats.shape

        if region_padding_mask is None:
            valid_mask = feats.abs().sum(dim=-1) > 0
        else:
            valid_mask = ~region_padding_mask

        lap = self._get_feat(feats, ("laplacian_norm", "edge_sharpness"))
        sharp = self._get_feat(feats, ("sharpness_norm", "nr_sharpness"))
        sat = self._get_feat(feats, ("saturation",))
        bright = self._get_feat(feats, ("brightness_mean", "brightness_uniformity"))
        over = self._get_feat(feats, ("over_expose", "over_exposure", "over_exposure_quality"))
        under = self._get_feat(feats, ("under_expose", "under_exposure"))

        score = torch.zeros(bsz, num_regions, device=device)
        weight_sum = torch.zeros(bsz, num_regions, device=device)

        if lap is not None:
            score = score + 0.30 * self._to_unit_interval(lap, "laplacian_norm")
            weight_sum = weight_sum + 0.30
        if sharp is not None:
            score = score + 0.30 * self._to_unit_interval(sharp, "sharpness_norm")
            weight_sum = weight_sum + 0.30
        if sat is not None:
            score = score + 0.10 * self._to_unit_interval(sat, "saturation")
            weight_sum = weight_sum + 0.10
        if bright is not None:
            bright_u = self._to_unit_interval(bright, "brightness_mean")
            bright_score = (1.0 - torch.abs(bright_u - 0.5) / 0.5).clamp(0.0, 1.0)
            score = score + 0.20 * bright_score
            weight_sum = weight_sum + 0.20
        if over is not None:
            over_u = self._to_unit_interval(over, "over_expose")
            over_score = over_u if "over_exposure_quality" in self.name_to_idx else (1.0 - over_u)
            score = score + 0.10 * over_score.clamp(0.0, 1.0)
            weight_sum = weight_sum + 0.10
        if under is not None:
            under_u = self._to_unit_interval(under, "under_expose")
            under_score = 1.0 - under_u
            score = score + 0.10 * under_score.clamp(0.0, 1.0)
            weight_sum = weight_sum + 0.10

        target = torch.where(weight_sum > 0, score / weight_sum.clamp_min(1e-6), torch.zeros_like(score))
        target = target.clamp(0.0, 1.0)
        target = target.masked_fill(~valid_mask, 0.0)
        return target

    def _pairwise_ranking_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        targets = targets.view(-1)
        if preds.numel() < 2:
            return torch.tensor(0.0, device=preds.device)

        diff_pred = preds.unsqueeze(1) - preds.unsqueeze(0)
        diff_gt = targets.unsqueeze(1) - targets.unsqueeze(0)
        valid = diff_gt.abs() > self.ranking_pair_threshold
        if not valid.any():
            return torch.tensor(0.0, device=preds.device)

        sign = torch.sign(diff_gt)
        loss = F.relu(self.ranking_margin - sign * diff_pred)
        return loss[valid].mean()

    def forward(self, out: Dict[str, torch.Tensor], target_scores: torch.Tensor, target_distributions: Optional[torch.Tensor] = None):
        final_score = out["final_score"].view(-1)
        region_scores = out.get("region_scores", None)
        region_scores_raw = out.get("region_scores_raw", region_scores)
        global_distribution = out.get("global_distribution", None)

        aggregated_region_attention_weights = out.get("aggregated_region_attention_weights", None)
        if aggregated_region_attention_weights is None:
            aggregated_region_attention_weights = out.get("region_weights", None)

        raw_region_attention_weights = out.get("region_attention_weights", None)
        head_importance = out.get("head_importance", None)
        region_padding_mask = out.get("region_padding_mask", None)
        deep_region_features = out.get("deep_region_features", None)

        hand_region_features = out.get("region_hand_align_feature", None)
        if hand_region_features is None:
            hand_region_features = out.get("handcrafted_region_features", None)

        hand_region_raw = out.get("handcrafted_region_raw", None)
        if hand_region_raw is None:
            hand_region_raw = out.get("raw_handcrafted_region", None)

        loss_global_raw = self._global_regression_loss(final_score, target_scores)
        loss_global = self.global_weight * loss_global_raw
        loss_rank = self._pairwise_ranking_loss(final_score, target_scores)

        loss_consistency = torch.tensor(0.0, device=final_score.device)
        loss_alignment = torch.tensor(0.0, device=final_score.device)
        loss_region_quality = torch.tensor(0.0, device=final_score.device)
        loss_distribution = torch.tensor(0.0, device=final_score.device)

        if global_distribution is not None and target_distributions is not None:
            loss_distribution = F.kl_div(torch.log(global_distribution + 1e-8), target_distributions, reduction="batchmean")

        attention_for_loss = aggregated_region_attention_weights if aggregated_region_attention_weights is not None else raw_region_attention_weights

        if region_scores_raw is not None:
            region_scores_for_loss = region_scores_raw
            if region_padding_mask is not None:
                region_scores_for_loss = region_scores_for_loss.masked_fill(region_padding_mask, 0.0)

            region_weights = self._compute_region_weights(
                region_scores=region_scores_for_loss,
                region_attention_weights=attention_for_loss,
                region_padding_mask=region_padding_mask,
                head_importance=head_importance,
            )
            region_agg_score = (region_scores_for_loss * region_weights).sum(dim=1)
            loss_consistency = F.smooth_l1_loss(region_agg_score, final_score.detach())

        if self.alignment_weight > 0.0 and deep_region_features is not None and hand_region_features is not None:
            if deep_region_features.shape[-1] != hand_region_features.shape[-1]:
                raise ValueError(
                    f"Alignment dim mismatch: deep_region_features={tuple(deep_region_features.shape)}, "
                    f"hand_region_features={tuple(hand_region_features.shape)}"
                )
            valid_mask = torch.ones(deep_region_features.shape[:2], dtype=torch.bool, device=deep_region_features.device) \
                if region_padding_mask is None else (~region_padding_mask)
            if valid_mask.any():
                diff = deep_region_features - hand_region_features
                diff = diff[valid_mask]
                loss_alignment = torch.mean(diff.pow(2))

        if self.region_quality_weight > 0.0:
            region_quality_target = self._build_region_quality_target(hand_region_raw, region_padding_mask)
            if (region_scores_raw is not None) and (region_quality_target is not None):
                valid_mask = torch.ones_like(region_scores_raw, dtype=torch.bool)
                if region_padding_mask is not None:
                    valid_mask = valid_mask & (~region_padding_mask)
                if valid_mask.any():
                    loss_region_quality = F.smooth_l1_loss(region_scores_raw[valid_mask], region_quality_target[valid_mask])

        total_loss = (
            loss_global
            + 0.5 * loss_distribution
            + self.ranking_weight * loss_rank
            + self.consistency_weight * loss_consistency
            + self.alignment_weight * loss_alignment
            + self.region_quality_weight * loss_region_quality
        )

        loss_dict = {
            "global": loss_global_raw.detach(),
            "rank": loss_rank.detach(),
            "dist": loss_distribution.detach(),
            "cons": loss_consistency.detach(),
            "align": loss_alignment.detach(),
            "region_quality": loss_region_quality.detach(),
        }
        return total_loss, loss_dict

# ============================================================
# 4. 工具函数
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


def compute_metrics(
    preds_np: np.ndarray,
    targets_np: np.ndarray,
    preds_scale: float = 1.0,
    targets_scale: float = 1.0,
    acc_tols: tuple = (0.05, 0.10),
    acc_names: tuple = ("acc@0.05", "acc@0.10"),
):
    if preds_np.shape[0] < 2:
        metrics = {"pearson": 0.0, "spearman": 0.0, "mse": 0.0, "mae": 0.0}
        for tol, name in zip(acc_tols, acc_names):
            metrics[name] = 0.0
        return metrics

    preds = preds_np.astype(float) * float(preds_scale)
    targets = targets_np.astype(float) * float(targets_scale)

    if preds.shape[0] < 2:
        pcc = 0.0
        srcc = 0.0
    else:
        if np.std(preds) < 1e-12 or np.std(targets) < 1e-12:
            pcc = 0.0
        else:
            try:
                pcc = float(np.corrcoef(preds, targets)[0, 1])
                if np.isnan(pcc):
                    pcc = 0.0
            except Exception:
                pcc = 0.0

        if len(np.unique(preds)) < 2 or len(np.unique(targets)) < 2:
            srcc = 0.0
        else:
            try:
                srcc = float(spearmanr(preds, targets).correlation)
                if np.isnan(srcc):
                    srcc = 0.0
            except Exception:
                srcc = 0.0

    diff = preds - targets
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(abs_diff))

    metrics = {"pearson": pcc, "spearman": srcc, "mse": mse, "mae": mae}
    for tol, name in zip(acc_tols, acc_names):
        metrics[name] = float(np.sum(abs_diff <= tol)) / len(abs_diff)

    return metrics


def compute_selection_score(metrics: Dict[str, float], srcc_weight: float = 0.80) -> float:
    srcc_weight = float(srcc_weight)
    pcc_weight = 1.0 - srcc_weight
    return srcc_weight * float(metrics["spearman"]) + pcc_weight * float(metrics["pearson"])


def compute_mean_std(results: List[Dict[str, float]]) -> Dict[str, dict]:
    """
    计算多次实验结果的平均值和标准差
    
    Args:
        results: 多个实验的指标字典列表
        
    Returns:
        包含每个指标的 mean 和 std 的字典
    """
    if len(results) == 0:
        return {}
    
    keys = results[0].keys()
    mean_std = {}
    
    for key in keys:
        values = [r[key] for r in results]
        mean_std[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values
        }
    
    return mean_std


def print_mean_std_results(mean_std: Dict[str, dict], title: str = "Ablation Results"):
    """
    打印平均值 ± 标准差格式的实验结果（确保小数点后四位完整显示）
    """
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    # 关键指标优先显示
    key_order = ["pearson", "spearman", "mse", "mae", "acc@0.05", "acc@0.10", "acc@0.5", "acc@1.0"]
    
    for key in key_order:
        if key in mean_std:
            m = mean_std[key]
            # 使用 format 确保四位小数完整显示，包括尾部的零
            mean_str = "{0:.4f}".format(m["mean"])
            std_str = "{0:.4f}".format(m["std"])
            print(f"  {key:12s}: {mean_str} ± {std_str}")
    
    # 显示剩余指标
    for key in mean_std.keys():
        if key not in key_order:
            m = mean_std[key]
            print(f"  {key:12s}: {m['mean']:.4f} ± {m['std']:.4f}")
    
    print(f"{'='*60}\n")


def _truncate_features_for_no_region(feats: Optional[torch.Tensor], global_feat_dim: int):
    if feats is None:
        return None
    return feats[:, :int(global_feat_dim)].clone()


def linear_schedule(start: float, end: float, idx: int, total: int) -> float:
    if total <= 1:
        return float(end)
    progress = float(idx) / float(max(total - 1, 1))
    return float(start + (end - start) * progress)


def describe_loss_weights(criterion: DesignAlignedLoss) -> str:
    return (
        f"rank={criterion.ranking_weight:.4f}, cons={criterion.consistency_weight:.4f}, "
        f"align={criterion.alignment_weight:.4f}, rq={criterion.region_quality_weight:.4f}"
    )


def build_phase_criterion(phase_name: str, phase_epoch_idx: int, phase_total_epochs: int) -> DesignAlignedLoss:
    phase_name = str(phase_name).lower()
    if phase_name == "warmup":
        return DesignAlignedLoss(
            global_loss_type=GLOBAL_LOSS_TYPE,
            global_loss_beta=GLOBAL_LOSS_BETA,
            global_weight=LOSS_GLOBAL_WEIGHT,
            consistency_weight=0.0,
            alignment_weight=0.0,
            region_quality_weight=0.0,
            ranking_weight=0.0,
            ranking_margin=RANKING_MARGIN,
            ranking_pair_threshold=RANKING_PAIR_THRESHOLD,
            feature_names=getattr(cfg, "REGION_FEATURE_NAMES", CSV_SUB_FEATS),
        ).to(DEVICE)

    if phase_name == "finetune":
        consistency_weight = linear_schedule(PHASE2_CONSISTENCY_START, PHASE2_CONSISTENCY_END, phase_epoch_idx, phase_total_epochs)
        alignment_weight = linear_schedule(PHASE2_ALIGNMENT_START, PHASE2_ALIGNMENT_END, phase_epoch_idx, phase_total_epochs)
        region_quality_weight = linear_schedule(PHASE2_REGION_QUALITY_START, PHASE2_REGION_QUALITY_END, phase_epoch_idx, phase_total_epochs)
        ranking_weight = linear_schedule(PHASE2_RANKING_START, PHASE2_RANKING_END, phase_epoch_idx, phase_total_epochs)
    elif phase_name == "resnet_unfreeze":
        consistency_weight = linear_schedule(PHASE3_CONSISTENCY_START, PHASE3_CONSISTENCY_END, phase_epoch_idx, phase_total_epochs)
        alignment_weight = linear_schedule(PHASE3_ALIGNMENT_START, PHASE3_ALIGNMENT_END, phase_epoch_idx, phase_total_epochs)
        region_quality_weight = linear_schedule(PHASE3_REGION_QUALITY_START, PHASE3_REGION_QUALITY_END, phase_epoch_idx, phase_total_epochs)
        ranking_weight = linear_schedule(PHASE3_RANKING_START, PHASE3_RANKING_END, phase_epoch_idx, phase_total_epochs)
    else:
        raise ValueError(f"Unknown phase_name: {phase_name}")

    return DesignAlignedLoss(
        global_loss_type=GLOBAL_LOSS_TYPE,
        global_loss_beta=GLOBAL_LOSS_BETA,
        global_weight=LOSS_GLOBAL_WEIGHT,
        consistency_weight=consistency_weight,
        alignment_weight=alignment_weight,
        region_quality_weight=region_quality_weight,
        ranking_weight=ranking_weight,
        ranking_margin=RANKING_MARGIN,
        ranking_pair_threshold=RANKING_PAIR_THRESHOLD,
        feature_names=getattr(cfg, "REGION_FEATURE_NAMES", CSV_SUB_FEATS),
    ).to(DEVICE)


def configure_vit_high_layers(vit_model, start_layer=9, verbose=True):
    for p in vit_model.parameters():
        p.requires_grad = False

    trainable_names = []
    target_prefixes = tuple(f"encoder.layer.{i}." for i in range(start_layer, 12))
    for name, p in vit_model.named_parameters():
        if name.startswith(target_prefixes) or name.startswith("layernorm"):
            p.requires_grad = True
            trainable_names.append(name)

    trainable_param_count = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    total_param_count = sum(p.numel() for p in vit_model.parameters())

    if verbose:
        print("\n🔍 [ViT Unfreeze Check]")
        print(f"   Start Layer: {start_layer}")
        print(f"   Trainable Params: {trainable_param_count / 1e6:.2f}M / {total_param_count / 1e6:.2f}M")
        print(f"   Trainable Tensors: {len(trainable_names)}")
        for n in trainable_names[:20]:
            print(f"      - {n}")
        if len(trainable_names) > 20:
            print(f"      ... ({len(trainable_names) - 20} more)")

    if trainable_param_count == 0:
        raise RuntimeError("ViT 高层解冻失败：没有任何参数被设为 requires_grad=True。")

    return trainable_names


def configure_resnet_trainable(model, mode: str = "layer4", verbose: bool = True):
    mode = str(mode).lower()
    backbone = getattr(model, "cnn_backbone", None)
    if backbone is None:
        return []

    for p in backbone.parameters():
        p.requires_grad = False

    if mode == "all":
        for p in backbone.parameters():
            p.requires_grad = True
    elif mode == "layer4":
        target_module = backbone[-1]
        for p in target_module.parameters():
            p.requires_grad = True
    elif mode == "none":
        pass
    else:
        raise ValueError(f"不支持的 RESNET_UNFREEZE_MODE: {mode}")

    trainable_names = []
    for name, p in model.named_parameters():
        if name.startswith("cnn_backbone") and p.requires_grad:
            trainable_names.append(name)

    if verbose:
        trainable_param_count = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        total_param_count = sum(p.numel() for p in backbone.parameters())
        print("\n🔍 [ResNet Unfreeze Check]")
        print(f"   Mode: {mode}")
        print(f"   Trainable Params: {trainable_param_count / 1e6:.2f}M / {total_param_count / 1e6:.2f}M")
        print(f"   Trainable Tensors: {len(trainable_names)}")
        for n in trainable_names[:20]:
            print(f"      - {n}")
        if len(trainable_names) > 20:
            print(f"      ... ({len(trainable_names) - 20} more)")
    return trainable_names


def load_checkpoint_for_model(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return ckpt
    model.load_state_dict(ckpt, strict=False)
    return {"epoch": "unknown", "val_metrics": None}


def _history_csv_path(save_name: str) -> str:
    stem = os.path.splitext(save_name)[0]
    return os.path.join(CHECKPOINT_DIR, f"{stem}_train_history.csv")


def _summary_json_path(save_name: str) -> str:
    stem = os.path.splitext(save_name)[0]
    return os.path.join(CHECKPOINT_DIR, f"{stem}_training_summary.json")


def _summary_txt_path(save_name: str) -> str:
    stem = os.path.splitext(save_name)[0]
    return os.path.join(CHECKPOINT_DIR, f"{stem}_training_summary.txt")


def save_history_csv(history_rows: List[Dict], save_name: str):
    if not history_rows:
        return
    path = _history_csv_path(save_name)
    fieldnames = sorted({k for row in history_rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)


def save_summary(summary: Dict, save_name: str):
    json_path = _summary_json_path(save_name)
    txt_path = _summary_txt_path(save_name)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join([f"{k}: {v}" for k, v in summary.items()]))


def current_lr_info(optimizer) -> Dict[str, float]:
    lrs = [pg["lr"] for pg in optimizer.param_groups]
    if not lrs:
        return {"lr_min": 0.0, "lr_max": 0.0}
    return {"lr_min": float(min(lrs)), "lr_max": float(max(lrs))}

# ============================================================
# 5. 训练结构：数据、参数冻结、优化器、调度器
# ============================================================
def build_dataloaders(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = get_feature_dataloader(
        split=TRAIN_SPLIT,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        generator=g,
        run_tag="default",
        strict_scaler=True,
    )
    val_loader = get_feature_dataloader(
        split=VAL_SPLIT,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        run_tag="default",
        strict_scaler=True,
    )
    test_loader = get_feature_dataloader(
        split=TEST_SPLIT,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        run_tag="default",
        strict_scaler=True,
    )
    return train_loader, val_loader, test_loader


def setup_trainable_params(model, phase: str):
    phase = str(phase).lower()

    if phase == "warmup":
        print("\n🔥 [Phase 1] Warmup (Freeze Backbones)...")
        for p in model.parameters():
            p.requires_grad = False

        backbone_modules = [getattr(model, "cnn_backbone", None), getattr(model, "vit", None), getattr(model, "seg_branch", None)]
        backbone_ids = set()
        for mod in backbone_modules:
            if mod is not None:
                for p in mod.parameters():
                    backbone_ids.add(id(p))

        p_heads = []
        for p in model.parameters():
            if id(p) not in backbone_ids:
                p.requires_grad = True
                p_heads.append(p)

        print(f"   ---> Activated {len(p_heads)} tensors for Warmup.")
        return {"phase": "warmup", "p_heads": p_heads, "p_vit": [], "p_resnet": [], "p_seg": []}

    if phase in {"finetune", "resnet_unfreeze"}:
        if phase == "finetune":
            print(f"\n🔓 [Phase 2] Selective Unfreeze (ViT High Layers + ResNet {PHASE2_RESNET_UNFREEZE_MODE})...")
            resnet_mode = PHASE2_RESNET_UNFREEZE_MODE
        else:
            print(f"\n🔓 [Phase 3] Unfreeze ResNet ({RESNET_UNFREEZE_MODE}) + Keep ViT High Layers...")
            resnet_mode = RESNET_UNFREEZE_MODE

        for p in model.parameters():
            p.requires_grad = False

        backbone_modules = [getattr(model, "cnn_backbone", None), getattr(model, "vit", None), getattr(model, "seg_branch", None)]
        backbone_ids = set()
        for mod in backbone_modules:
            if mod is not None:
                for p in mod.parameters():
                    backbone_ids.add(id(p))
        for p in model.parameters():
            if id(p) not in backbone_ids:
                p.requires_grad = True

        if getattr(model, "cnn_backbone", None) is not None:
            configure_resnet_trainable(model, mode=resnet_mode, verbose=True)

        if getattr(model, "vit", None) is not None:
            configure_vit_high_layers(model.vit, start_layer=VIT_UNFREEZE_START_LAYER, verbose=True)

        if getattr(model, "seg_branch", None) is not None:
            for p in model.seg_branch.parameters():
                p.requires_grad = (not SEG_FREEZE)

        if FREEZE_BATCHNORM_STATS and getattr(model, "cnn_backbone", None) is not None:
            for m in model.cnn_backbone.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

        p_vit = [p for p in model.vit.parameters() if p.requires_grad] if getattr(model, "vit", None) is not None else []
        p_resnet = [p for p in model.cnn_backbone.parameters() if p.requires_grad] if getattr(model, "cnn_backbone", None) is not None else []
        p_seg = [p for p in model.seg_branch.parameters() if p.requires_grad] if getattr(model, "seg_branch", None) is not None else []

        backbone_trainable_ids = set(id(p) for p in (p_vit + p_resnet + p_seg))
        p_heads = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_trainable_ids]

        print(f"   ---> Params Groups: ViT={len(p_vit)}, ResNet={len(p_resnet)}, Seg={len(p_seg)}, Heads={len(p_heads)}")
        return {"phase": phase, "p_heads": p_heads, "p_vit": p_vit, "p_resnet": p_resnet, "p_seg": p_seg}

    raise ValueError(f"不支持的训练阶段: {phase}")


def build_optimizer(model, phase: str):
    group_info = setup_trainable_params(model, phase=phase)

    if phase == "warmup":
        optimizer = optim.AdamW(
            group_info["p_heads"],
            lr=PHASE1_HEAD_LR,
            weight_decay=WEIGHT_DECAY,
            betas=(BETA1, BETA2),
            eps=EPS,
        )
        return optimizer, group_info

    head_lr = PHASE2_HEAD_LR if phase == "finetune" else PHASE3_HEAD_LR
    vit_lr = PHASE2_VIT_LR if phase == "finetune" else PHASE3_VIT_LR
    resnet_lr = PHASE2_RESNET_LR if phase == "finetune" else PHASE3_RESNET_LR

    param_groups = []
    if group_info["p_vit"] and vit_lr > 0:
        param_groups.append({"params": group_info["p_vit"], "lr": vit_lr, "weight_decay": BACKBONE_WEIGHT_DECAY, "name": "vit"})
    if group_info["p_resnet"] and resnet_lr > 0:
        param_groups.append({"params": group_info["p_resnet"], "lr": resnet_lr, "weight_decay": BACKBONE_WEIGHT_DECAY, "name": "resnet"})
    if group_info["p_heads"] and head_lr > 0:
        param_groups.append({"params": group_info["p_heads"], "lr": head_lr, "weight_decay": WEIGHT_DECAY, "name": "heads"})
    if group_info["p_seg"] and SEG_LR_SCALE > 0:
        param_groups.append({"params": group_info["p_seg"], "lr": LR * SEG_LR_SCALE, "weight_decay": BACKBONE_WEIGHT_DECAY, "name": "seg"})

    for pg in param_groups:
        print(f"   ---> ParamGroup[{pg.get('name', 'group')}] lr={pg['lr']:.2e} wd={pg['weight_decay']:.2e} count={len(pg['params'])}")

    optimizer = optim.AdamW(param_groups, betas=(BETA1, BETA2), eps=EPS)
    return optimizer, group_info


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1, initial_lr_ratio=0.1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = max(int(total_epochs), 1)
        self.eta_min = eta_min
        self.initial_lr_ratio = initial_lr_ratio
        self.optimizer = optimizer
        self.initial_lrs = [group['lr'] * self.initial_lr_ratio for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                self.initial_lrs[i] + (base_lr - self.initial_lrs[i]) * (self.last_epoch + 1) / max(self.warmup_epochs, 1)
                for i, base_lr in enumerate(self.base_lrs)
            ]
        progress = (self.last_epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2 for base_lr in self.base_lrs]


def build_scheduler(optimizer, scheduler_type: str, total_epochs: int):
    scheduler_type = str(scheduler_type).lower()
    if scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=PLATEAU_FACTOR,
            patience=PLATEAU_PATIENCE,
            threshold=PLATEAU_THRESHOLD,
            min_lr=MIN_LR,
        )
    if scheduler_type == "cosine":
        warmup_epochs = min(2, max(1, total_epochs // 3))
        return WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            eta_min=MIN_LR,
            initial_lr_ratio=0.2,
        )
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE,
        threshold=PLATEAU_THRESHOLD,
        min_lr=MIN_LR,
    )

# ============================================================
# 6. 数据解析与 forward 封装
# ============================================================
def _extract_batch_tensors(batch, ablation_type: str, global_feat_dim: int):
    imgs = batch["image"].to(DEVICE).float()
    vit_imgs = batch.get("vit_image", batch["image"]).to(DEVICE).float()
    seg_imgs = batch.get("seg_image", batch["image"]).to(DEVICE).float()

    semantic_mask = batch.get("semantic_mask", None)
    if semantic_mask is not None:
        semantic_mask = semantic_mask.to(DEVICE).long()

    feats = batch.get("features", None)
    if feats is not None:
        feats = feats.to(DEVICE).float()

    if ablation_type in {"vision_only", "no_handcrafted_features"}:
        feats = None
    elif ablation_type == "no_region_features":
        feats = _truncate_features_for_no_region(feats, global_feat_dim)

    region_valid = batch.get("region_valid", None)
    region_area_ratio = batch.get("region_area_ratio", None)
    if region_valid is not None:
        region_valid = region_valid.to(DEVICE).float()
    if region_area_ratio is not None:
        region_area_ratio = region_area_ratio.to(DEVICE).float()

    score_distribution = None
    return imgs, vit_imgs, seg_imgs, semantic_mask, feats, region_valid, region_area_ratio, score_distribution


def _split_handcrafted_features(features, region_meta=None):
    if features is None:
        return None, None

    if features.dim() == 1:
        features = features.unsqueeze(0)

    batch_size = features.size(0)
    expected_region_core_dim = cfg.REGION_HANDCRAFTED_DIM - cfg.REGION_META_PER_REGION
    expected_total_dim = cfg.GLOBAL_CORE_DIM + cfg.REGION_COUNT * expected_region_core_dim

    if features.size(1) != expected_total_dim:
        raise ValueError(
            f"Handcrafted feature dim mismatch: got {features.size(1)}, expected {expected_total_dim} "
            f"(GLOBAL_CORE_DIM={cfg.GLOBAL_CORE_DIM}, REGION_COUNT={cfg.REGION_COUNT}, "
            f"REGION_HANDCRAFTED_DIM={cfg.REGION_HANDCRAFTED_DIM}, REGION_META_PER_REGION={cfg.REGION_META_PER_REGION})"
        )

    global_handcrafted = features[:, :cfg.GLOBAL_CORE_DIM]
    region_core = features[:, cfg.GLOBAL_CORE_DIM:].view(batch_size, cfg.REGION_COUNT, expected_region_core_dim)

    if region_meta is None:
        region_meta = torch.zeros(
            batch_size,
            cfg.REGION_COUNT,
            cfg.REGION_META_PER_REGION,
            device=features.device,
            dtype=features.dtype,
        )
    elif region_meta.dim() == 2:
        region_meta = region_meta.view(batch_size, cfg.REGION_COUNT, cfg.REGION_META_PER_REGION)

    expected_meta_shape = (batch_size, cfg.REGION_COUNT, cfg.REGION_META_PER_REGION)
    if tuple(region_meta.shape) != expected_meta_shape:
        raise ValueError(f"region_meta shape mismatch: got {tuple(region_meta.shape)}, expected {expected_meta_shape}")

    region_handcrafted = torch.cat([region_core, region_meta], dim=-1)
    if region_handcrafted.size(-1) != cfg.REGION_HANDCRAFTED_DIM:
        raise ValueError(f"Region handcrafted dim mismatch: got {region_handcrafted.size(-1)}, expected {cfg.REGION_HANDCRAFTED_DIM}")

    return global_handcrafted, region_handcrafted


def _forward_model(
    model,
    imgs,
    vit_imgs,
    seg_imgs,
    semantic_mask,
    feats,
    region_valid,
    region_area_ratio,
    score_distribution=None,
    scores=None,
    mode="infer",
):
    region_meta = None
    if region_valid is not None and region_area_ratio is not None:
        if region_valid.dim() == 2 and region_area_ratio.dim() == 2:
            region_meta = torch.stack([region_valid, region_area_ratio], dim=-1)
        elif region_valid.dim() == 3 and region_area_ratio.dim() == 3:
            region_meta = torch.cat([region_valid, region_area_ratio], dim=-1)
        else:
            region_meta = torch.cat(
                [
                    region_valid.view(region_valid.size(0), -1, 1),
                    region_area_ratio.view(region_area_ratio.size(0), -1, 1),
                ],
                dim=-1,
            )

    handcrafted_global, handcrafted_region = _split_handcrafted_features(feats, region_meta=region_meta)
    seg_masks = semantic_mask if USE_OFFLINE_SEG_MASK else None

    return model(
        x=imgs,
        x_vit=vit_imgs,
        handcrafted_global=handcrafted_global,
        handcrafted_region=handcrafted_region,
        seg_masks=seg_masks,
    )

# ============================================================
# 7. 验证与测试
# ============================================================
@torch.no_grad()
def validate(model, loader, criterion=None, ablation_type="full", global_feat_dim=7, desc="[Val]"):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    valid_batches = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        imgs, vit_imgs, seg_imgs, semantic_mask, feats, region_valid, region_area_ratio, score_distribution = _extract_batch_tensors(
            batch, ablation_type=ablation_type, global_feat_dim=global_feat_dim
        )
        y = batch["label"].to(DEVICE).float().view(-1)

        out = _forward_model(
            model, imgs, vit_imgs, seg_imgs, semantic_mask, feats, region_valid, region_area_ratio,
            score_distribution=score_distribution, scores=None, mode="infer"
        )
        pred_scores = out["final_score"].view(-1)

        if criterion is None:
            loss = F.smooth_l1_loss(pred_scores, y, beta=GLOBAL_LOSS_BETA) if GLOBAL_LOSS_TYPE != "mse" else F.mse_loss(pred_scores, y)
        else:
            loss, _ = criterion(out, y, target_distributions=None)

        total_loss += loss.item()
        valid_batches += 1
        all_preds.append(pred_scores.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets)) if len(all_preds) else compute_metrics(np.zeros(2), np.zeros(2))
    return metrics, total_loss / max(valid_batches, 1)


def test_evaluate(model, loader, ablation_type="full", global_feat_dim=7, use_tta=True, num_tta=5, save_path: Optional[str] = None):
    model.eval()
    all_preds, all_targets = [], []
    all_image_ids = []

    if use_tta and not bool(getattr(model.cfg, "use_online_handcrafted_extractor", False)):
        print("⚠️ Disable TTA because offline handcrafted features are not augmented consistently.")
        use_tta = False

    print("🚀 Running Inference...")
    if use_tta:
        print(f"🔄 Using Test Time Augmentation (TTA) with {num_tta} augmentations")

    for batch in tqdm(loader, desc="[Test]", leave=False):
        imgs, vit_imgs, seg_imgs, semantic_mask, feats, region_valid, region_area_ratio, score_distribution = _extract_batch_tensors(
            batch, ablation_type=ablation_type, global_feat_dim=global_feat_dim
        )

        if "label_raw" in batch:
            y = batch["label_raw"].to(DEVICE).float().view(-1)
            acc_tols = (0.5, 1.0)
            acc_names = ("acc@0.5", "acc@1.0")
        else:
            y = batch["label"].to(DEVICE).float().view(-1)
            acc_tols = (0.05, 0.10)
            acc_names = ("acc@0.05", "acc@0.10")

        if use_tta:
            tta_preds = []
            out = _forward_model(model, imgs, vit_imgs, seg_imgs, semantic_mask, feats, region_valid, region_area_ratio,
                                 score_distribution=score_distribution, scores=None, mode="infer")
            tta_preds.append(out["final_score_10"].view(-1))

            for _ in range(num_tta - 1):
                imgs_flipped = torch.flip(imgs, dims=[3])
                vit_imgs_flipped = torch.flip(vit_imgs, dims=[3])
                seg_imgs_flipped = torch.flip(seg_imgs, dims=[3]) if seg_imgs is not None and seg_imgs.dim() >= 4 else seg_imgs
                semantic_mask_flipped = torch.flip(semantic_mask, dims=[2]) if semantic_mask is not None and semantic_mask.dim() >= 3 else semantic_mask

                out = _forward_model(
                    model, imgs_flipped, vit_imgs_flipped, seg_imgs_flipped, semantic_mask_flipped,
                    feats, region_valid, region_area_ratio, score_distribution=score_distribution, scores=None, mode="infer"
                )
                tta_preds.append(out["final_score_10"].view(-1))

            pred_scores = torch.stack(tta_preds, dim=0).mean(dim=0)
        else:
            out = _forward_model(model, imgs, vit_imgs, seg_imgs, semantic_mask, feats, region_valid, region_area_ratio,
                                 score_distribution=score_distribution, scores=None, mode="infer")
            pred_scores = out["final_score_10"].view(-1)

        all_preds.append(pred_scores.detach().cpu().numpy())
        all_targets.append((y.detach().cpu().numpy() * 10) if "label_raw" not in batch else y.detach().cpu().numpy())

        if "image_id" in batch:
            ids = batch["image_id"]
            if isinstance(ids, list):
                all_image_ids.extend([str(x) for x in ids])
            else:
                all_image_ids.extend([str(x.item()) if hasattr(x, "item") else str(x) for x in ids])
        else:
            bs = pred_scores.shape[0]
            start = sum(len(a) for a in all_preds[:-1])
            all_image_ids.extend([f"idx_{start + j}" for j in range(bs)])

    preds_np = np.concatenate(all_preds) if len(all_preds) else np.array([])
    targets_np = np.concatenate(all_targets) if len(all_targets) else np.array([])

    metrics = compute_metrics(preds_np, targets_np, acc_tols=acc_tols, acc_names=acc_names) if len(all_preds) else compute_metrics(np.zeros(2), np.zeros(2), acc_tols=acc_tols, acc_names=acc_names)

    if save_path and len(all_image_ids) == len(preds_np):
        import os as _os
        import pandas as _pd
        _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
        df = _pd.DataFrame({
            "image_id": all_image_ids,
            "label_raw": targets_np,
            "pred_raw": preds_np,
        })
        df.to_csv(save_path, index=False)
        print(f"💾 Per-sample predictions saved: {save_path} ({len(df)} rows)")

    return metrics, {"preds": preds_np, "targets": targets_np, "image_ids": all_image_ids}

# ============================================================
# 8. 训练
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, epoch, config):
    model.train()

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.eval()

    total_loss_accum = 0.0
    total_global = total_rank = total_dist = total_cons = total_align = total_rq = 0.0
    total_region_weights_mean = 0.0
    total_region_weights_std = 0.0
    total_region_score_valid_std = 0.0
    valid_batches = 0

    all_preds, all_targets = [], []
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Train]", dynamic_ncols=True, smoothing=0.0, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")
    for i, batch in enumerate(pbar):
        imgs, vit_imgs, seg_imgs, semantic_mask, feats, region_valid, region_area_ratio, score_distribution = _extract_batch_tensors(
            batch, ablation_type=config["ablation_type"], global_feat_dim=config["global_feat_dim"]
        )
        scores = batch["label"].to(DEVICE).float().view(-1)

        out = _forward_model(
            model, imgs, vit_imgs, seg_imgs, semantic_mask, feats, region_valid, region_area_ratio,
            score_distribution=score_distribution, scores=scores, mode="train"
        )
        pred_scores = out["final_score"].view(-1)

        loss, loss_dict = criterion(out, scores, target_distributions=None)
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ 检测到 NaN/Inf loss，跳过当前 batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        (loss / config["accumulation_steps"]).backward()

        if (i + 1) % config["accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss_accum += loss.item()
        total_global += loss_dict["global"].item()
        total_rank += loss_dict["rank"].item()
        total_dist += loss_dict["dist"].item()
        total_cons += loss_dict["cons"].item()
        total_align += loss_dict["align"].item()
        total_rq += loss_dict["region_quality"].item()
        valid_batches += 1

        all_preds.append(pred_scores.detach().cpu().numpy())
        all_targets.append(scores.detach().cpu().numpy())

        region_scores_raw = out.get("region_scores_raw", None)
        region_padding_mask = out.get("region_padding_mask", None)
        region_weights = out.get("region_weights", None)
        if region_weights is not None:
            safe_weights = region_weights
            if region_padding_mask is not None:
                safe_weights = safe_weights.masked_fill(region_padding_mask, 0.0)
            total_region_weights_mean += float(safe_weights.mean().item())
            total_region_weights_std += float(safe_weights.std().item())

        if region_scores_raw is not None:
            if region_padding_mask is not None:
                valid_mask = ~region_padding_mask
                if valid_mask.any():
                    total_region_score_valid_std += float(region_scores_raw[valid_mask].std().item())
            else:
                total_region_score_valid_std += float(region_scores_raw.std().item())

        if i % 5 == 0:
            batch_metrics = compute_metrics(pred_scores.detach().cpu().numpy(), scores.detach().cpu().numpy())
            postfix = {
                "Total": f"{loss.item():.4f}",
                "G": f"{loss_dict['global'].item():.4f}",
                "R": f"{loss_dict['rank'].item():.4f}",
                "C": f"{loss_dict['cons'].item():.4f}",
                "A": f"{loss_dict['align'].item():.4f}",
                "RQ": f"{loss_dict['region_quality'].item():.4f}",
                "SRCC": f"{batch_metrics['spearman']:.4f}",
                "PCC": f"{batch_metrics['pearson']:.4f}",
            }
            if region_scores_raw is not None:
                if region_padding_mask is not None:
                    valid_mask = ~region_padding_mask
                    valid_regions_mean = float(valid_mask.float().sum(dim=1).mean().item())
                    safe_scores = region_scores_raw.masked_fill(~valid_mask, 0.0)
                    region_std = float(safe_scores.std().item())
                else:
                    valid_regions_mean = float(region_scores_raw.size(1))
                    region_std = float(region_scores_raw.std().item())
                postfix["ValidR"] = f"{valid_regions_mean:.2f}"
                postfix["RStd"] = f"{region_std:.4f}"
            if region_weights is not None:
                postfix["WMean"] = f"{float(region_weights.mean().item()):.4f}"
                postfix["WStd"] = f"{float(region_weights.std().item()):.4f}"

            pbar.set_postfix(postfix)

    if len(loader) % config["accumulation_steps"] != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets)) if len(all_preds) > 0 else compute_metrics(np.zeros(2), np.zeros(2))

    print(
        f"📊 [Train] Epoch {epoch + 1}: Loss={total_loss_accum / max(valid_batches, 1):.4f} | "
        f"G={total_global / max(valid_batches, 1):.4f} | R={total_rank / max(valid_batches, 1):.4f} | "
        f"D={total_dist / max(valid_batches, 1):.4f} | C={total_cons / max(valid_batches, 1):.4f} | "
        f"A={total_align / max(valid_batches, 1):.4f} | RQ={total_rq / max(valid_batches, 1):.4f} | "
        f"WMean={total_region_weights_mean / max(valid_batches, 1):.4f} | "
        f"WStd={total_region_weights_std / max(valid_batches, 1):.4f} | "
        f"RScoreStd={total_region_score_valid_std / max(valid_batches, 1):.4f} | "
        f"SRCC={metrics['spearman']:.4f} | PCC={metrics['pearson']:.4f}"
    )

    return total_loss_accum / max(valid_batches, 1), metrics, {
        "global": total_global / max(valid_batches, 1),
        "rank": total_rank / max(valid_batches, 1),
        "dist": total_dist / max(valid_batches, 1),
        "cons": total_cons / max(valid_batches, 1),
        "align": total_align / max(valid_batches, 1),
        "region_quality": total_rq / max(valid_batches, 1),
        "region_weights_mean": total_region_weights_mean / max(valid_batches, 1),
        "region_weights_std": total_region_weights_std / max(valid_batches, 1),
        "region_score_valid_std": total_region_score_valid_std / max(valid_batches, 1),
    }

# ============================================================
# 9. 只测试，不训练
# ============================================================
def run_test_only(save_name="Latest_SSG_CAF_model.pth", ablation_type="full"):
    print(f"📦 Loading Test Set (Split: {TEST_SPLIT})...")
    _, _, test_loader = build_dataloaders(seed=SEED)

    print("🏗️ Initializing Model...")
    model = build_full_model(DEVICE, ablation_type=ablation_type)
    if cfg.PRINT_MODEL_INFO:
        print_system_check(model)

    ckpt_path = os.path.join(CHECKPOINT_DIR, save_name)
    ckpt = load_checkpoint_for_model(model, ckpt_path, DEVICE)

    print(f"✅ Loaded checkpoint: {ckpt_path}")
    print(f"   ↳ Saved epoch: {ckpt.get('epoch', 'unknown')}")
    if ckpt.get("val_metrics") is not None:
        print(f"   ↳ Saved val metrics: {ckpt['val_metrics']}")

    use_tta = bool(getattr(model.cfg, "use_online_handcrafted_extractor", False))

    test_pred_csv = os.path.join(CHECKPOINT_DIR, "predictions", f"SSG_CAF_{ablation_type}_seed{SEED}_test_predictions.csv")
    os.makedirs(os.path.dirname(test_pred_csv), exist_ok=True)

    test_m, _ = test_evaluate(
        model,
        test_loader,
        ablation_type=ablation_type,
        global_feat_dim=GLOBAL_FEAT_DIM,
        use_tta=use_tta,
        num_tta=5 if use_tta else 1,
        save_path=test_pred_csv,
    )

    print(
        f"🏆 Final Results: PLCC={test_m['pearson']:.4f} SRCC={test_m['spearman']:.4f} "
        f"MSE={test_m['mse']:.4f} MAE={test_m['mae']:.4f}"
    )

# ============================================================
# 10. 主训练流程
# ============================================================
def restore_best_checkpoint_if_exists(model, save_path: str, tag: str):
    if os.path.exists(save_path):
        ckpt = load_checkpoint_for_model(model, save_path, DEVICE)
        print(f"🔁 [{tag}] 已恢复 best checkpoint (epoch={ckpt.get('epoch', 'unknown')})")
        return ckpt
    print(f"ℹ️ [{tag}] 未找到可恢复的 best checkpoint: {save_path}")
    return None


def train_model_logic(
    save_name="Latest_SSG_CAF_model.pth",
    seed=SEED,
    ablation_type="full",
    early_stop_patience=None,
    scheduler_type: str = SCHEDULER_TYPE_DEFAULT,
    start_phase=1,
):
    global WEIGHT_DECAY, BACKBONE_WEIGHT_DECAY
    global PHASE3_HEAD_LR, PHASE3_VIT_LR, PHASE3_RESNET_LR

    # ============================================================
    # 🎲 随机种子检查
    # ============================================================
    print(f"\n🎲 [Seed Check] 设置随机种子: {seed}")
    # 生成一个随机数用于验证
    import random
    random_check = random.random()
    print(f"   随机数验证: {random_check:.6f}")
    print(f"   如果种子正确，不同seed的随机数应该不同")
    print("="*50)
    # ============================================================
    
    set_global_seed(seed)
    save_path = os.path.join(CHECKPOINT_DIR, save_name)
    history_rows: List[Dict] = []

    model = build_full_model(DEVICE, ablation_type=ablation_type)
    if cfg.PRINT_MODEL_INFO:
        print_system_check(model)

    config = {
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "early_stop_patience": early_stop_patience if early_stop_patience is not None else EARLY_STOP_PATIENCE,
        "ablation_type": ablation_type,
        "global_feat_dim": GLOBAL_FEAT_DIM,
    }

    train_loader, val_loader, test_loader = build_dataloaders(seed=seed)

    best_selection_score = -1.0
    best_srcc = -1.0
    best_epoch = -1
    best_val_metrics = None
    early_stop_counter = 0
    phase2_improved = False

    # ========================================================
    # Phase 1: Warmup
    # ========================================================
    if start_phase == 1:
        optimizer, _ = build_optimizer(model, phase="warmup")
        scheduler = build_scheduler(optimizer, scheduler_type=scheduler_type, total_epochs=WARMUP_EPOCHS)
        criterion_warmup = build_phase_criterion("warmup", 0, max(WARMUP_EPOCHS, 1))

        for epoch in range(WARMUP_EPOCHS):
            print(f"\n🧪 Warmup Loss Weights: {describe_loss_weights(criterion_warmup)}")
            train_loss, train_metrics, train_loss_items = train_one_epoch(
                model, train_loader, optimizer, criterion_warmup, epoch, {"epochs": WARMUP_EPOCHS, **config}
            )

            val_metrics, val_loss = validate(
                model, val_loader, criterion=criterion_warmup,
                ablation_type=ablation_type, global_feat_dim=config["global_feat_dim"]
            )
            val_selection_score = compute_selection_score(val_metrics, srcc_weight=BEST_METRIC_SRCC_WEIGHT)

            lr_info = current_lr_info(optimizer)
            row = {
                "phase": "warmup",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_srcc": train_metrics["spearman"],
                "train_pcc": train_metrics["pearson"],
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "val_srcc": val_metrics["spearman"],
                "val_pcc": val_metrics["pearson"],
                "val_selection_score": val_selection_score,
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                **train_loss_items,
                **lr_info,
            }
            history_rows.append(row)
            save_history_csv(history_rows, save_name)

            print(
                f"   [Val] Loss={val_loss:.4f} | SRCC={val_metrics['spearman']:.4f} | PCC={val_metrics['pearson']:.4f} | "
                f"Sel={val_selection_score:.4f} | MSE={val_metrics['mse']:.4f} | MAE={val_metrics['mae']:.4f}"
            )

            improved = val_selection_score > best_selection_score + 1e-4
            if improved:
                best_selection_score = val_selection_score
                best_srcc = val_metrics["spearman"]
                best_epoch = epoch + 1
                best_val_metrics = val_metrics
                early_stop_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_metrics": val_metrics,
                    "val_loss": val_loss,
                    "val_selection_score": val_selection_score,
                    "epoch": epoch + 1,
                    "history_csv": _history_csv_path(save_name),
                }, save_path)
                print("   🏆 New Best Saved")
            else:
                early_stop_counter += 1
    else:
        if os.path.exists(save_path):
            print(f"📦 Loading saved model from {save_path}...")
            ckpt = torch.load(save_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"✅ Loaded model from epoch {ckpt.get('epoch', 'unknown')}")
            best_selection_score = ckpt.get("val_selection_score", -1.0)
            best_srcc = best_selection_score
            best_epoch = ckpt.get("epoch", WARMUP_EPOCHS if start_phase == 2 else WARMUP_EPOCHS + PHASE2_EPOCHS)
            best_val_metrics = ckpt.get("val_metrics", None)
            print(f"📊 Restored best selection score: {best_selection_score:.4f}")
        else:
            print(f"⚠️ No saved model found. Starting Phase {start_phase} from scratch.")
            best_selection_score = -1.0
            best_srcc = -1.0
            best_epoch = WARMUP_EPOCHS if start_phase == 2 else WARMUP_EPOCHS + PHASE2_EPOCHS
            best_val_metrics = None

    if start_phase <= 2 and start_phase == 1:
        restore_best_checkpoint_if_exists(model, save_path, tag="Before Phase2")

    # ========================================================
    # Phase 2: Selective Finetuning
    # ========================================================
    local_best_before_phase2 = best_selection_score
    if start_phase <= 2 and PHASE2_EPOCHS > 0:
        optimizer_phase2, _ = build_optimizer(model, phase="finetune")
        scheduler_phase2 = build_scheduler(optimizer_phase2, scheduler_type=scheduler_type, total_epochs=PHASE2_EPOCHS)
        early_stop_counter = 0

        for phase2_idx in range(PHASE2_EPOCHS):
            epoch = WARMUP_EPOCHS + phase2_idx
            criterion_finetune = build_phase_criterion("finetune", phase2_idx, PHASE2_EPOCHS)
            print(f"\n🧪 Phase2 Loss Weights: {describe_loss_weights(criterion_finetune)}")

            train_loss, train_metrics, train_loss_items = train_one_epoch(
                model, train_loader, optimizer_phase2, criterion_finetune, epoch,
                {"epochs": WARMUP_EPOCHS + PHASE2_EPOCHS, **config}
            )

            val_metrics, val_loss = validate(
                model, val_loader, criterion=criterion_finetune,
                ablation_type=ablation_type, global_feat_dim=config["global_feat_dim"]
            )

            val_selection_score = compute_selection_score(val_metrics, srcc_weight=BEST_METRIC_SRCC_WEIGHT)
            if scheduler_type == "plateau":
                scheduler_phase2.step(val_selection_score)
            else:
                scheduler_phase2.step()

            lr_info = current_lr_info(optimizer_phase2)
            row = {
                "phase": "finetune",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_srcc": train_metrics["spearman"],
                "train_pcc": train_metrics["pearson"],
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "val_srcc": val_metrics["spearman"],
                "val_pcc": val_metrics["pearson"],
                "val_selection_score": val_selection_score,
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                **train_loss_items,
                **lr_info,
            }
            history_rows.append(row)
            save_history_csv(history_rows, save_name)

            print(
                f"✨ [Val] Ep {epoch + 1}: SRCC={val_metrics['spearman']:.4f} | PCC={val_metrics['pearson']:.4f} | "
                f"Sel={val_selection_score:.4f} | Loss={val_loss:.4f} | MSE={val_metrics['mse']:.4f} | MAE={val_metrics['mae']:.4f}"
            )

            improved = val_selection_score > best_selection_score + 1e-4
            if improved:
                best_selection_score = val_selection_score
                best_srcc = val_metrics["spearman"]
                best_epoch = epoch + 1
                best_val_metrics = val_metrics
                early_stop_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_metrics": val_metrics,
                    "val_loss": val_loss,
                    "val_selection_score": val_selection_score,
                    "epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_srcc": best_srcc,
                    "history_csv": _history_csv_path(save_name),
                }, save_path)
                print("   🏆 New Best Saved")
            else:
                early_stop_counter += 1
                print(f"   ⏳ No improve: {early_stop_counter}/{config['early_stop_patience']}")
                if early_stop_counter >= config["early_stop_patience"]:
                    print("⏹️ Early stopping triggered in phase2.")
                    break

    if start_phase <= 2:
        phase2_improved = best_selection_score > local_best_before_phase2 + 1e-4
        restore_best_checkpoint_if_exists(model, save_path, tag="Before Phase3")
    else:
        phase2_improved = True

    # ========================================================
    # Phase 3: Unfreeze ResNet layer4
    # ========================================================
    if PHASE3_EPOCHS > 0 and (not PHASE3_ONLY_IF_PHASE2_IMPROVED or phase2_improved):
        print("\n📉 [Phase 3] Using reduced learning rates and stronger region supervision")

        original_phase3_head_lr = PHASE3_HEAD_LR
        original_phase3_vit_lr = PHASE3_VIT_LR
        original_phase3_resnet_lr = PHASE3_RESNET_LR
        original_backbone_wd = BACKBONE_WEIGHT_DECAY
        original_head_wd = WEIGHT_DECAY

        PHASE3_HEAD_LR = float(getattr(cfg, "PHASE3_HEAD_LR", original_phase3_head_lr))
        PHASE3_VIT_LR = float(getattr(cfg, "PHASE3_VIT_LR", original_phase3_vit_lr))
        PHASE3_RESNET_LR = float(getattr(cfg, "PHASE3_RESNET_LR", original_phase3_resnet_lr))
        BACKBONE_WEIGHT_DECAY = float(getattr(cfg, "BACKBONE_WEIGHT_DECAY_PHASE3", original_backbone_wd))
        WEIGHT_DECAY = float(getattr(cfg, "HEAD_WEIGHT_DECAY_PHASE3", original_head_wd))

        optimizer_phase3, _ = build_optimizer(model, phase="resnet_unfreeze")
        scheduler_phase3 = build_scheduler(optimizer_phase3, scheduler_type=scheduler_type, total_epochs=PHASE3_EPOCHS)
        early_stop_counter = 0

        for phase3_idx in range(PHASE3_EPOCHS):
            epoch = WARMUP_EPOCHS + PHASE2_EPOCHS + phase3_idx
            criterion_phase3 = build_phase_criterion("resnet_unfreeze", phase3_idx, PHASE3_EPOCHS)
            print(f"\n🧪 Phase3 Loss Weights: {describe_loss_weights(criterion_phase3)}")

            train_loss, train_metrics, train_loss_items = train_one_epoch(
                model, train_loader, optimizer_phase3, criterion_phase3, epoch,
                {"epochs": WARMUP_EPOCHS + PHASE2_EPOCHS + PHASE3_EPOCHS, **config}
            )

            val_metrics, val_loss = validate(
                model, val_loader, criterion=criterion_phase3,
                ablation_type=ablation_type, global_feat_dim=config["global_feat_dim"]
            )

            val_selection_score = compute_selection_score(val_metrics, srcc_weight=BEST_METRIC_SRCC_WEIGHT)
            if scheduler_type == "plateau":
                scheduler_phase3.step(val_selection_score)
            else:
                scheduler_phase3.step()

            lr_info = current_lr_info(optimizer_phase3)
            row = {
                "phase": "resnet_unfreeze",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_srcc": train_metrics["spearman"],
                "train_pcc": train_metrics["pearson"],
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "val_srcc": val_metrics["spearman"],
                "val_pcc": val_metrics["pearson"],
                "val_selection_score": val_selection_score,
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                **train_loss_items,
                **lr_info,
            }
            history_rows.append(row)
            save_history_csv(history_rows, save_name)

            print(
                f"✨ [Val] Ep {epoch + 1}: SRCC={val_metrics['spearman']:.4f} | PCC={val_metrics['pearson']:.4f} | "
                f"Sel={val_selection_score:.4f} | Loss={val_loss:.4f} | MSE={val_metrics['mse']:.4f} | MAE={val_metrics['mae']:.4f}"
            )

            improved = val_selection_score > best_selection_score + 1e-4
            if improved:
                best_selection_score = val_selection_score
                best_srcc = val_metrics["spearman"]
                best_epoch = epoch + 1
                best_val_metrics = val_metrics
                early_stop_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer_phase2.state_dict(),
                    "scheduler_state_dict": scheduler_phase2.state_dict(),
                    "val_metrics": val_metrics,
                    "val_loss": val_loss,
                    "val_selection_score": val_selection_score,
                    "epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_srcc": best_srcc,
                    "history_csv": _history_csv_path(save_name),
                }, save_path)
                print("   🏆 New Best Saved")
            else:
                early_stop_counter += 1
                print(f"   ⏳ No improve: {early_stop_counter}/{config['early_stop_patience']}")
                if early_stop_counter >= config["early_stop_patience"]:
                    print("⏹️ Early stopping triggered.")
                    break

        PHASE3_HEAD_LR = original_phase3_head_lr
        PHASE3_VIT_LR = original_phase3_vit_lr
        PHASE3_RESNET_LR = original_phase3_resnet_lr
        BACKBONE_WEIGHT_DECAY = original_backbone_wd
        WEIGHT_DECAY = original_head_wd
    elif PHASE3_EPOCHS > 0:
        print("\n⏭️ [Phase 3] 跳过：phase2 未带来稳定收益，不解冻 ResNet。")

    # ========================================================
    # Final Test
    # ========================================================
    print("\n🏁 Final Evaluation on Test Set.")
    test_metrics = None

    if os.path.exists(save_path):
        ckpt = load_checkpoint_for_model(model, save_path, DEVICE)
        print(f"✅ Loaded best checkpoint from epoch: {ckpt.get('epoch', 'unknown')}")

        test_pred_csv = os.path.join(CHECKPOINT_DIR, "predictions", f"SSG_CAF_full_seed{seed}_test_predictions.csv")
        os.makedirs(os.path.dirname(test_pred_csv), exist_ok=True)

        test_metrics, _ = test_evaluate(
            model,
            test_loader,
            ablation_type=config["ablation_type"],
            global_feat_dim=config["global_feat_dim"],
            save_path=test_pred_csv,
        )

        if "acc@0.5" in test_metrics:
            print(
                f"🏆 Final Results: PLCC={test_metrics['pearson']:.4f} SRCC={test_metrics['spearman']:.4f} "
                f"MSE={test_metrics['mse']:.4f} MAE={test_metrics['mae']:.4f} "
                f"ACC@0.5={test_metrics['acc@0.5']:.4f} ACC@1.0={test_metrics['acc@1.0']:.4f}"
            )
        else:
            print(
                f"🏆 Final Results: PLCC={test_metrics['pearson']:.4f} SRCC={test_metrics['spearman']:.4f} "
                f"MSE={test_metrics['mse']:.4f} MAE={test_metrics['mae']:.4f} "
                f"ACC@0.05={test_metrics['acc@0.05']:.4f} ACC@0.10={test_metrics['acc@0.10']:.4f}"
            )

    summary = {
        "save_path": save_path,
        "best_epoch": best_epoch,
        "best_srcc": best_srcc,
        "best_selection_score": best_selection_score,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "history_csv": _history_csv_path(save_name),
        "seed": seed,
        "ablation_type": ablation_type,
        "scheduler_type": scheduler_type,
        "warmup_epochs": WARMUP_EPOCHS,
        "phase2_epochs": PHASE2_EPOCHS,
        "phase3_epochs": PHASE3_EPOCHS,
    }
    save_summary(summary, save_name)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--save_name", type=str, default="SSG_CAF_model_region_balanced_v4.pth")
    parser.add_argument("--ablation_type", type=str, default="full")
    parser.add_argument("--scheduler", type=str, default=SCHEDULER_TYPE_DEFAULT, choices=["cosine", "plateau"])
    parser.add_argument("--config_module", type=str, default=CONFIG_MODULE)
    parser.add_argument("--start_phase", type=int, default=1, choices=[1, 2, 3], help="Start training from which phase")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for ablation experiments (42, 123, 2024)")
    args = parser.parse_args()

    if args.test_only:
        run_test_only(save_name=args.save_name, ablation_type=args.ablation_type)
    else:
        train_model_logic(
            save_name=args.save_name,
            seed=args.seed,
            ablation_type=args.ablation_type,
            scheduler_type=args.scheduler,
            start_phase=args.start_phase,
        )
