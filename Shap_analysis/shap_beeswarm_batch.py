# shap_beeswarm_batch.py
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

try:
    import shap  # type: ignore
except Exception:
    shap = None


# =========================================================
# Project setup
# =========================================================
def project_setup() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd()]
    for cand in candidates:
        if (cand / "utils").exists() and (cand / "models").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    return cwd


PROJECT_ROOT = project_setup()

import utils.config as cfg  # noqa: E402
from utils.config import ModelConfig  # noqa: E402
from models.full_model import FullAestheticModel  # noqa: E402

try:
    from utils.data_loader import AestheticFeatureDataset  # noqa: E402
except Exception:
    AestheticFeatureDataset = None


# =========================================================
# Constants
# =========================================================
DEFAULT_REGION_NAMES = [
    "background", "face", "sky", "building", "vegetation", "water", "object", "noise"
]
META_FEATURES = {"valid", "area_ratio"}

REGION_EN = {
    "background": "Background",
    "face": "Face",
    "sky": "Sky",
    "building": "Building",
    "vegetation": "Vegetation",
    "water": "Water",
    "object": "Object",
    "noise": "Noise",
}

FEATURE_EN = {
    "edge_sharpness": "Edge sharpness",
    "nr_sharpness": "NR sharpness",
    "color_cast": "Color cast",
    "saturation": "Saturation",
    "color_balance": "Color balance",
    "over_exposure": "Over-exposure quality",
    "under_exposure": "Under-exposure quality",
    "brightness_uniformity": "Brightness uniformity",
    "composition": "Composition",
    "symmetry": "Symmetry",
    "texture": "Texture",
    "contrast": "Contrast",
    "valid": "Region validity",
    "area_ratio": "Area ratio",
}

# Lower-saturation styling closer to a paper figure.
COLOR_LOW = "#2A7FDB"
COLOR_HIGH = "#D81B60"
BAR_BG = "#EED2DE"
FIG_BG = "#FFFFFF"
GRID_COLOR = "#CFCFD4"
ZERO_LINE = "#8D8D94"
AXIS_COLOR = "#3F3F46"
TEXT_COLOR = "#2E2E33"
SHAP_CMAP = LinearSegmentedColormap.from_list("shap_like_soft", [COLOR_LOW, "#8A56C5", COLOR_HIGH])


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Naming helpers
# =========================================================
def region_display_name_en(region_name: str) -> str:
    return REGION_EN.get(region_name, region_name)


def global_feature_display_name_en(feature_name: str) -> str:
    return FEATURE_EN.get(feature_name, feature_name)


def plot_label_from_feature_name(raw_name: str) -> str:
    raw_name = str(raw_name)
    if raw_name.startswith("global_"):
        base_feat = raw_name[len("global_"):]
        return f"Global {FEATURE_EN.get(base_feat, base_feat)}"
    for region_name in resolve_region_names():
        prefix = f"{region_name}_"
        if raw_name.startswith(prefix):
            base_feat = raw_name[len(prefix):]
            return f"{region_display_name_en(region_name)} {FEATURE_EN.get(base_feat, base_feat)}"
    return raw_name


def shorten_plot_label(label: str, max_len: int = 26) -> str:
    return label if len(label) <= max_len else label[: max_len - 1] + "…"


# =========================================================
# Config resolution
# =========================================================
def resolve_region_names() -> List[str]:
    labels = getattr(cfg, "SEMANTIC_LABELS", None)
    if isinstance(labels, dict):
        try:
            return [str(labels[i]) for i in sorted(labels.keys())]
        except Exception:
            return [str(v) for _, v in sorted(labels.items())]
    if isinstance(labels, (list, tuple)):
        return [str(x) for x in labels]
    return list(DEFAULT_REGION_NAMES)


def resolve_global_feature_names() -> List[str]:
    names = getattr(cfg, "GLOBAL_FEATURE_NAMES", None)
    if isinstance(names, (list, tuple)) and len(names) > 0:
        return [str(x) for x in names]
    core = getattr(cfg, "CORE_FEATURE_ORDER", None)
    if isinstance(core, (list, tuple)) and len(core) > 0:
        return [str(x) for x in core]
    global_dim = int(getattr(cfg, "GLOBAL_CORE_DIM", 12))
    return [f"global_feat_{i}" for i in range(global_dim)]


def resolve_region_feature_names() -> List[str]:
    names = getattr(cfg, "REGION_FEATURE_NAMES", None)
    if isinstance(names, (list, tuple)) and len(names) > 0:
        return [str(x) for x in names]
    core = list(getattr(cfg, "CORE_FEATURE_ORDER", [f"region_feat_{i}" for i in range(12)]))
    meta = list(getattr(cfg, "CSV_SUB_FEATS", ["valid", "area_ratio"]))
    return [str(x) for x in (core + meta)]


# =========================================================
# Model / data helpers
# =========================================================
def safe_torch_load(
    path: str | os.PathLike[str],
    map_location: str | torch.device = "cpu",
    *,
    weights_only: bool = False,
) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model(checkpoint_path: str, device: torch.device) -> FullAestheticModel:
    model_cfg = ModelConfig()
    model_cfg.region_type_embed_dim = 15
    model = FullAestheticModel(model_cfg).to(device)

    ckpt = safe_torch_load(checkpoint_path, map_location=device, weights_only=False)
    raw_state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt

    model_state = model.state_dict()
    adapted_state = {}
    for k, v in raw_state_dict.items():
        if k not in model_state:
            continue
        target = model_state[k]
        if tuple(v.shape) == tuple(target.shape):
            adapted_state[k] = v
            continue
        if (
            k == "prediction_module.region_score_head.net.0.weight"
            and v.ndim == 2
            and target.ndim == 2
            and v.shape[0] == target.shape[0]
            and v.shape[1] + 1 == target.shape[1]
        ):
            new_weight = target.clone()
            new_weight.zero_()
            new_weight[:, : v.shape[1]] = v
            adapted_state[k] = new_weight
            continue

    model.load_state_dict(adapted_state, strict=False)
    model.eval()
    return model


def _split_handcrafted_features(
    features: torch.Tensor,
    region_valid: Optional[torch.Tensor],
    region_area_ratio: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if features.dim() == 1:
        features = features.unsqueeze(0)

    batch_size = features.size(0)
    global_core_dim = int(getattr(cfg, "GLOBAL_CORE_DIM", len(resolve_global_feature_names())))
    region_count = int(getattr(cfg, "REGION_COUNT", getattr(cfg, "SEG_NUM_CLASSES", 8)))
    region_meta_per_region = int(getattr(cfg, "REGION_META_PER_REGION", 2))
    region_handcrafted_dim = int(getattr(cfg, "REGION_HANDCRAFTED_DIM", len(resolve_region_feature_names())))
    region_core_dim = region_handcrafted_dim - region_meta_per_region

    global_handcrafted = features[:, :global_core_dim]
    region_core = features[:, global_core_dim:].view(batch_size, region_count, region_core_dim)

    if region_valid is None:
        region_valid = torch.zeros(batch_size, region_count, device=features.device, dtype=features.dtype)
    if region_area_ratio is None:
        region_area_ratio = torch.zeros(batch_size, region_count, device=features.device, dtype=features.dtype)

    region_meta = torch.stack([region_valid, region_area_ratio], dim=-1)
    region_handcrafted = torch.cat([region_core, region_meta], dim=-1)
    return global_handcrafted, region_handcrafted


def prepare_sample_inputs(sample: Dict[str, Any], device: torch.device, use_online_seg: bool = False) -> Dict[str, Any]:
    image = sample["image"].unsqueeze(0).to(device).float()
    vit_image = sample.get("vit_image", sample["image"]).unsqueeze(0).to(device).float()

    semantic_mask = sample.get("semantic_mask", None)
    if semantic_mask is not None and not use_online_seg:
        semantic_mask = semantic_mask.unsqueeze(0).to(device).long()
    else:
        semantic_mask = None

    feats = sample["features"].unsqueeze(0).to(device).float()
    region_valid = sample.get("region_valid", None)
    region_area_ratio = sample.get("region_area_ratio", None)
    if region_valid is not None:
        region_valid = region_valid.unsqueeze(0).to(device).float()
    if region_area_ratio is not None:
        region_area_ratio = region_area_ratio.unsqueeze(0).to(device).float()

    handcrafted_global, handcrafted_region = _split_handcrafted_features(feats, region_valid, region_area_ratio)

    return {
        "x": image,
        "x_vit": vit_image,
        "seg_masks": semantic_mask,
        "handcrafted_global": handcrafted_global,
        "handcrafted_region": handcrafted_region,
        "region_valid": region_valid,
        "region_area_ratio": region_area_ratio,
        "image_id": str(sample.get("image_id", "unknown")),
        "label": float(sample["label"].item() if torch.is_tensor(sample["label"]) else sample["label"]),
    }


@torch.no_grad()
def run_model_forward(model: FullAestheticModel, sample_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    return model(
        x=sample_inputs["x"],
        x_vit=sample_inputs["x_vit"],
        handcrafted_global=sample_inputs["handcrafted_global"],
        handcrafted_region=sample_inputs["handcrafted_region"],
        seg_masks=sample_inputs["seg_masks"],
    )


def build_background(cache_path: Optional[str], max_samples: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    if cache_path and os.path.isfile(cache_path):
        payload = safe_torch_load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "global" in payload and "region" in payload:
            return payload["global"].float(), payload["region"].float()

    if AestheticFeatureDataset is None:
        raise RuntimeError("AestheticFeatureDataset is unavailable.")

    dataset = AestheticFeatureDataset(
        split=getattr(cfg, "TRAIN_SPLIT", "train"),
        run_tag="default",
        strict_scaler=True,
    )

    bg_global: List[torch.Tensor] = []
    bg_region: List[torch.Tensor] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        feats = sample["features"].unsqueeze(0).float()
        region_valid = sample.get("region_valid", None)
        region_area_ratio = sample.get("region_area_ratio", None)
        if region_valid is not None:
            region_valid = region_valid.unsqueeze(0).float()
        if region_area_ratio is not None:
            region_area_ratio = region_area_ratio.unsqueeze(0).float()
        g, r = _split_handcrafted_features(feats, region_valid, region_area_ratio)
        bg_global.append(g.squeeze(0))
        bg_region.append(r.squeeze(0))
        if len(bg_global) >= max_samples:
            break

    out_g = torch.stack(bg_global, dim=0)
    out_r = torch.stack(bg_region, dim=0)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"global": out_g, "region": out_r}, cache_path)
    return out_g, out_r


def manual_forward_from_intermediate_features(
    model: FullAestheticModel,
    outputs: Dict[str, torch.Tensor],
    handcrafted_global: torch.Tensor,
    handcrafted_region: torch.Tensor,
    apply_region_dropout: bool = False,
) -> Dict[str, Any]:
    if hasattr(model, "forward_from_intermediate_features"):
        return model.forward_from_intermediate_features(
            global_deep_feature=outputs["deep_global_feature"],
            region_deep_features=outputs.get("deep_region_features"),
            patch_tokens=outputs["patch_tokens"],
            handcrafted_global=handcrafted_global,
            handcrafted_region=handcrafted_region,
            region_padding_mask=outputs.get("region_padding_mask"),
            image_region_soft_masks=outputs.get("image_region_soft_masks"),
            region_soft_masks=outputs.get("region_soft_masks"),
            apply_region_dropout=apply_region_dropout,
        )
    raise RuntimeError("Model lacks forward_from_intermediate_features interface.")


# =========================================================
# SHAP wrappers / engine
# =========================================================
class _ScoreWrapper(nn.Module):
    def __init__(
        self,
        full_model: nn.Module,
        outputs_cache: Dict[str, torch.Tensor],
        global_dim: int,
        num_regions: int,
        region_dim: int,
        manual_forward_fn,
        target_score: str = "final",
    ) -> None:
        super().__init__()
        self.full_model = full_model
        self.outputs_cache = outputs_cache
        self.global_dim = int(global_dim)
        self.num_regions = int(num_regions)
        self.region_dim = int(region_dim)
        self.manual_forward_fn = manual_forward_fn
        self.target_score = target_score

    def _maybe_expand(self, value: Any, batch_size: int) -> Any:
        if value is None:
            return None
        if not torch.is_tensor(value):
            return value
        if value.dim() > 0 and value.size(0) == 1 and batch_size > 1:
            expand_sizes = [batch_size] + [-1] * (value.dim() - 1)
            return value.expand(*expand_sizes).clone()
        return value.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = self.outputs_cache["deep_global_feature"].device
        x = x.to(device)
        handcrafted_global = x[:, :self.global_dim]
        handcrafted_region = x[:, self.global_dim:].reshape(-1, self.num_regions, self.region_dim)

        batch_size = handcrafted_global.size(0)
        outputs_cache: Dict[str, Any] = {k: self._maybe_expand(v, batch_size) for k, v in self.outputs_cache.items()}

        out = self.manual_forward_fn(
            self.full_model,
            outputs_cache,
            handcrafted_global,
            handcrafted_region,
            apply_region_dropout=False,
        )
        key = "global_score_10" if self.target_score == "global" else "final_score_10"
        y = out[key]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        elif y.ndim > 2:
            y = y.reshape(y.shape[0], -1)[:, :1]
        return y


class BatchShapEngine:
    def __init__(
        self,
        region_names: Sequence[str],
        global_feature_names: Sequence[str],
        region_feature_names: Sequence[str],
        manual_forward_fn,
        target_score: str = "final",
    ) -> None:
        self.region_names = list(region_names)
        self.global_feature_names = list(global_feature_names)
        self.region_feature_names = list(region_feature_names)
        self.manual_forward_fn = manual_forward_fn
        self.target_score = target_score

    def _flatten_features(self, hg: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return torch.cat([hg, hr.flatten(start_dim=1)], dim=1)

    def compute_shap_for_sample(
        self,
        model: nn.Module,
        outputs: Dict[str, torch.Tensor],
        background_global: torch.Tensor,
        background_region: torch.Tensor,
        sample_global: torch.Tensor,
        sample_region: torch.Tensor,
    ) -> Tuple[np.ndarray, float, str, np.ndarray]:
        wrapper = _ScoreWrapper(
            full_model=model,
            outputs_cache=outputs,
            global_dim=sample_global.size(-1),
            num_regions=sample_region.size(1),
            region_dim=sample_region.size(2),
            manual_forward_fn=self.manual_forward_fn,
            target_score=self.target_score,
        ).eval()

        device = outputs["deep_global_feature"].device
        bg = self._flatten_features(background_global.to(device), background_region.to(device))
        x = self._flatten_features(sample_global.to(device), sample_region.to(device))

        if shap is None:
            x_grad = x.clone().detach().requires_grad_(True)
            y = wrapper(x_grad).sum()
            y.backward()
            vals = x_grad.grad.detach().cpu().numpy() * x.detach().cpu().numpy()
            base_value = float(wrapper(bg).mean().detach().cpu().item())
            shap_vec = np.asarray(vals[0], dtype=np.float32).reshape(-1)
            x_vec = np.asarray(x.detach().cpu().numpy()[0], dtype=np.float32).reshape(-1)
            return shap_vec, base_value, "Grad*Input", x_vec

        try:
            explainer = shap.GradientExplainer(wrapper, bg)
            shap_values = explainer.shap_values(x)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_arr = np.asarray(shap_values, dtype=np.float32)
            expected_value = getattr(explainer, "expected_value", None)
            if isinstance(expected_value, (list, tuple, np.ndarray)):
                base_value = float(np.asarray(expected_value).reshape(-1)[0])
            elif expected_value is None:
                base_value = float(wrapper(bg).mean().detach().cpu().item())
            else:
                base_value = float(expected_value)
            shap_vec = np.asarray(shap_arr[0], dtype=np.float32).reshape(-1)
            x_vec = np.asarray(x.detach().cpu().numpy()[0], dtype=np.float32).reshape(-1)
            return shap_vec, base_value, "GradientExplainer", x_vec
        except Exception:
            x_grad = x.clone().detach().requires_grad_(True)
            y = wrapper(x_grad).sum()
            y.backward()
            vals = x_grad.grad.detach().cpu().numpy() * x.detach().cpu().numpy()
            base_value = float(wrapper(bg).mean().detach().cpu().item())
            shap_vec = np.asarray(vals[0], dtype=np.float32).reshape(-1)
            x_vec = np.asarray(x.detach().cpu().numpy()[0], dtype=np.float32).reshape(-1)
            return shap_vec, base_value, "Grad*Input", x_vec


# =========================================================
# Stratified sampling
# =========================================================
def extract_labels_and_ids(dataset: Any) -> Tuple[np.ndarray, List[str]]:
    if hasattr(dataset, "df"):
        df = getattr(dataset, "df")
        try:
            if "mean_score" in df.columns:
                labels = df["mean_score"].astype(float).to_numpy()
            elif "label" in df.columns:
                labels = df["label"].astype(float).to_numpy()
            else:
                labels = None
        except Exception:
            labels = None
        try:
            if "image_id" in df.columns:
                image_ids = [str(x) for x in df["image_id"].tolist()]
            else:
                image_ids = []
        except Exception:
            image_ids = []
        if labels is not None and len(image_ids) == len(labels) and len(labels) == len(dataset):
            return labels, image_ids

    labels_list: List[float] = []
    image_ids: List[str] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        label_val = sample.get("label")
        if torch.is_tensor(label_val):
            label_val = float(label_val.item())
        else:
            label_val = float(label_val)
        labels_list.append(label_val)
        image_ids.append(str(sample.get("image_id", idx)))
    return np.asarray(labels_list, dtype=np.float32), image_ids


def proportional_stratified_sample_indices(
    labels: np.ndarray,
    sample_size: int,
    num_bins: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(labels)
    if sample_size >= n:
        return np.arange(n), np.zeros(n, dtype=np.int64)

    bin_edges = np.linspace(1.0, 10.0, num_bins + 1, dtype=np.float32)
    bin_ids = np.digitize(labels, bin_edges[1:-1], right=False)

    rng = np.random.default_rng(seed)
    counts = np.bincount(bin_ids, minlength=num_bins).astype(np.int64)
    raw_targets = counts / counts.sum() * sample_size
    target_counts = np.floor(raw_targets).astype(np.int64)
    target_counts = np.minimum(target_counts, counts)

    remaining = int(sample_size - target_counts.sum())
    if remaining > 0:
        fractional = raw_targets - np.floor(raw_targets)
        candidates = np.argsort(-fractional)
        for b in candidates:
            if remaining <= 0:
                break
            if target_counts[b] < counts[b]:
                target_counts[b] += 1
                remaining -= 1

    if remaining > 0:
        candidates = np.argsort(-counts)
        for b in candidates:
            while remaining > 0 and target_counts[b] < counts[b]:
                target_counts[b] += 1
                remaining -= 1
            if remaining <= 0:
                break

    selected: List[int] = []
    for b in range(num_bins):
        idxs = np.where(bin_ids == b)[0]
        k = int(target_counts[b])
        if k <= 0 or len(idxs) == 0:
            continue
        chosen = rng.choice(idxs, size=k, replace=False)
        selected.extend(chosen.tolist())

    selected = np.asarray(sorted(selected), dtype=np.int64)
    return selected, bin_ids


# =========================================================
# Plotting helpers
# =========================================================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _subsample_for_plot(
    shap_row: np.ndarray,
    val_row: np.ndarray,
    max_points: int = 280,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    shap_row = np.asarray(shap_row, dtype=np.float32).reshape(-1)
    val_row = np.asarray(val_row, dtype=np.float32).reshape(-1)

    n = len(shap_row)
    if n <= max_points:
        return shap_row, val_row

    rng = np.random.default_rng(seed)
    order = np.argsort(shap_row, kind="mergesort")
    shap_sorted = shap_row[order]
    val_sorted = val_row[order]

    keep_idx = np.linspace(0, n - 1, max_points, dtype=int)
    keep_idx = np.clip(keep_idx + rng.integers(-1, 2, size=max_points), 0, n - 1)
    keep_idx = np.unique(keep_idx)
    if len(keep_idx) < max_points:
        remaining = np.setdiff1d(np.arange(n), keep_idx)
        extra = rng.choice(remaining, size=max_points - len(keep_idx), replace=False)
        keep_idx = np.sort(np.concatenate([keep_idx, extra]))

    return shap_sorted[keep_idx], val_sorted[keep_idx]


def _beeswarm_offsets_soft(
    x: np.ndarray,
    max_spread: float = 0.18,
    nbins: int = 30,
    seed: int = 42,
) -> np.ndarray:
    if len(x) == 0:
        return np.asarray([], dtype=np.float32)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if math.isclose(xmin, xmax):
        return np.zeros(len(x), dtype=np.float32)

    bins = np.linspace(xmin, xmax, nbins + 1)
    bin_ids = np.digitize(x, bins[1:-1], right=False)
    offsets = np.zeros(len(x), dtype=np.float32)
    rng = np.random.default_rng(seed)

    for b in np.unique(bin_ids):
        idx = np.where(bin_ids == b)[0]
        n = len(idx)
        if n <= 1:
            offsets[idx] = 0.0
            continue
        base = np.linspace(-max_spread, max_spread, n, dtype=np.float32)
        rng.shuffle(base)
        base += rng.normal(0.0, max_spread * 0.04, size=n)
        offsets[idx] = base
    return offsets


def save_beeswarm_importance_plot(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    save_path: str,
    top_k: int = 8,
    title: str = "SHAP feature importance and beeswarm distribution",
    max_points_per_feature: int = 280,
) -> List[Dict[str, Any]]:
    shap_values = np.asarray(shap_values, dtype=np.float32)
    feature_values = np.asarray(feature_values, dtype=np.float32)

    if shap_values.ndim != 2:
        raise ValueError(f"shap_values should be 2D [N,F], got shape={shap_values.shape}")
    if feature_values.ndim != 2:
        raise ValueError(f"feature_values should be 2D [N,F], got shape={feature_values.shape}")
    if shap_values.shape != feature_values.shape:
        raise ValueError(
            f"Shape mismatch: shap_values.shape={shap_values.shape}, feature_values.shape={feature_values.shape}"
        )

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    total_abs = np.sum(mean_abs) + 1e-12
    top_idx = np.argsort(mean_abs)[::-1][:top_k]

    shap_top = shap_values[:, top_idx]
    value_top = feature_values[:, top_idx]
    name_top = [str(feature_names[i]) for i in top_idx]
    label_top = [shorten_plot_label(plot_label_from_feature_name(n)) for n in name_top]
    mean_abs_top = mean_abs[top_idx]
    pct_top = mean_abs_top / total_abs * 100.0
    label_top_with_value = [f"{label} ({m:.4f})" for label, m in zip(label_top, mean_abs_top)]

    nfeat = len(name_top)
    y_positions = np.arange(nfeat)[::-1]

    # Use robust axis limits for a less stretched figure.
    max_abs_shap = float(np.percentile(np.abs(shap_top), 99.2))
    x_max = max(0.10, math.ceil(max_abs_shap / 0.025) * 0.025)
    x_min = -x_max
    mean_max = float(np.max(mean_abs_top) * 1.08)
    if mean_max <= 0:
        mean_max = 1.0

    # Give the color bar its own axis so it stays visually separated from the main plot.
    fig = plt.figure(figsize=(9.6, 6.2), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[40, 1.2], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    ax.set_facecolor(FIG_BG)
    cax.set_facecolor("white")

    plot_width = x_max - x_min
    for y, m in zip(y_positions, mean_abs_top):
        bar_w = (m / mean_max) * plot_width
        ax.barh(
            y,
            width=bar_w,
            left=x_min,
            height=0.58,
            color=BAR_BG,
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )

    for row_i, y in enumerate(y_positions):
        shap_row_full = shap_top[:, row_i]
        val_row_full = value_top[:, row_i]
        shap_row, val_row = _subsample_for_plot(
            shap_row_full,
            val_row_full,
            max_points=max_points_per_feature,
            seed=42 + row_i,
        )

        order = np.argsort(shap_row, kind="mergesort")
        shap_row = shap_row[order]
        val_row = val_row[order]
        offsets = _beeswarm_offsets_soft(shap_row, max_spread=0.16, nbins=24, seed=123 + row_i)
        y_row = np.full_like(shap_row, fill_value=float(y), dtype=np.float32) + offsets

        v_low = float(np.percentile(val_row_full, 5))
        v_high = float(np.percentile(val_row_full, 95))
        if math.isclose(v_low, v_high):
            normed = np.full_like(val_row, 0.5, dtype=np.float32)
        else:
            normed = np.clip((val_row - v_low) / (v_high - v_low), 0.0, 1.0)

        colors = SHAP_CMAP(normed)
        ax.scatter(
            shap_row,
            y_row,
            s=14,
            c=colors,
            edgecolors="none",
            alpha=0.82,
            zorder=3,
        )

    ax.axvline(0.0, color=ZERO_LINE, lw=1.6, alpha=0.9, zorder=1)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.linspace(x_min, x_max, 5))
    ax.grid(axis="y", color=GRID_COLOR, linestyle=(0, (1, 4)), linewidth=0.9, alpha=0.85)
    ax.set_axisbelow(True)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(label_top_with_value, fontsize=11)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax.tick_params(axis="x", labelsize=11, length=5, width=1.0, colors=AXIS_COLOR)
    ax.tick_params(axis="y", length=0, colors=AXIS_COLOR)

    ax_top = ax.twiny()
    ax_top.set_xlim(0.0, mean_max)
    ax_top.set_xticks(np.linspace(0.0, mean_max, 5))
    ax_top.set_xlabel("Mean  (|SHAP|  value)", fontsize=10)
    ax_top.tick_params(axis="x", labelsize=10, length=4, width=0.9, colors=AXIS_COLOR)

    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = ScalarMappable(norm=norm, cmap=SHAP_CMAP)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=11, length=0, colors=AXIS_COLOR)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.outline.set_visible(False)
    cbar.set_label("Feature value", fontsize=12, rotation=90, labelpad=12)

    # Make the color bar read as a separate side column.
    cax.spines["left"].set_visible(True)
    cax.spines["left"].set_color("#DDDDDD")
    cax.spines["left"].set_linewidth(0.8)
    for spine in ["top", "right", "bottom"]:
        cax.spines[spine].set_visible(False)

    ax.set_title(title, fontsize=15.5, pad=8, fontweight="normal", color=TEXT_COLOR)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color(AXIS_COLOR)
    ax_top.spines["top"].set_linewidth(1.0)
    ax_top.spines["top"].set_color(AXIS_COLOR)
    for spine in ["bottom", "left", "right"]:
        ax_top.spines[spine].set_visible(False)

    # Avoid tight_layout / bbox_inches='tight' so the reserved gap for the color bar is preserved.
    fig.subplots_adjust(left=0.34, right=0.95, top=0.90, bottom=0.12)
    plt.savefig(save_path, format="svg", dpi=260, facecolor=fig.get_facecolor())
    plt.close(fig)

    summary_rows: List[Dict[str, Any]] = []
    for n, label, m, p in zip(name_top, label_top, mean_abs_top, pct_top):
        summary_rows.append({
            "feature_name": n,
            "feature_display_name": label,
            "mean_abs_shap": float(m),
            "percentage": float(p),
        })
    return summary_rows


# =========================================================
# Batch execution
# =========================================================
def _build_region_flat_feature_names(
    region_names: Sequence[str],
    region_feature_names: Sequence[str],
) -> List[str]:
    out: List[str] = []
    for region_name in region_names:
        for feat_name in region_feature_names:
            if feat_name in META_FEATURES:
                continue
            out.append(f"{region_name}_{feat_name}")
    return out


def _extract_region_vectors(
    shap_flat: np.ndarray,
    flat_input: np.ndarray,
    global_dim: int,
    region_names: Sequence[str],
    region_feature_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    region_dim = len(region_feature_names)
    shap_out: List[float] = []
    val_out: List[float] = []
    for ridx, _ in enumerate(region_names):
        start = global_dim + ridx * region_dim
        for fidx, feat_name in enumerate(region_feature_names):
            if feat_name in META_FEATURES:
                continue
            idx = start + fidx
            shap_out.append(float(shap_flat[idx]))
            val_out.append(float(flat_input[idx]))
    return np.asarray(shap_out, dtype=np.float32), np.asarray(val_out, dtype=np.float32)


def explain_batch_and_plot(
    *,
    dataset: Any,
    selected_indices: Sequence[int],
    model: nn.Module,
    bg_global: torch.Tensor,
    bg_region: torch.Tensor,
    engine: BatchShapEngine,
    device: torch.device,
    global_feature_names: Sequence[str],
    region_names: Sequence[str],
    region_feature_names: Sequence[str],
    save_dir: str,
    use_online_seg: bool,
    progress_every: int = 25,
    top_k_global: int = 8,
    top_k_region: int = 8,
) -> Dict[str, Any]:
    _ensure_dir(save_dir)

    global_dim = len(global_feature_names)
    global_shap_rows: List[np.ndarray] = []
    global_feat_rows: List[np.ndarray] = []
    region_shap_rows: List[np.ndarray] = []
    region_feat_rows: List[np.ndarray] = []
    selected_meta: List[Dict[str, Any]] = []
    shap_method = None

    for i, idx in enumerate(selected_indices, start=1):
        sample = dataset[int(idx)]
        prepared = prepare_sample_inputs(sample, device, use_online_seg=use_online_seg)
        outputs = run_model_forward(model, prepared)

        shap_flat, _, method_name, flat_input = engine.compute_shap_for_sample(
            model=model,
            outputs=outputs,
            background_global=bg_global,
            background_region=bg_region,
            sample_global=prepared["handcrafted_global"],
            sample_region=prepared["handcrafted_region"],
        )
        shap_method = method_name
        global_shap_rows.append(np.asarray(shap_flat[:global_dim], dtype=np.float32).reshape(-1))
        global_feat_rows.append(np.asarray(flat_input[:global_dim], dtype=np.float32).reshape(-1))
        region_shap_vec, region_feat_vec = _extract_region_vectors(
            shap_flat=shap_flat,
            flat_input=flat_input,
            global_dim=global_dim,
            region_names=region_names,
            region_feature_names=region_feature_names,
        )
        region_shap_rows.append(region_shap_vec)
        region_feat_rows.append(region_feat_vec)

        pred_key = "global_score_10" if engine.target_score == "global" else "final_score_10"
        selected_meta.append({
            "dataset_index": int(idx),
            "image_id": prepared["image_id"],
            "true_score": float(prepared["label"]),
            "pred_score": float(outputs[pred_key][0].item()),
        })

        if i % progress_every == 0 or i == len(selected_indices):
            print(f"[progress] explained {i}/{len(selected_indices)} samples")

    global_shap_matrix = np.vstack(global_shap_rows)
    global_feat_matrix = np.vstack(global_feat_rows)
    region_shap_matrix = np.vstack(region_shap_rows)
    region_feat_matrix = np.vstack(region_feat_rows)

    global_summary_rows = save_beeswarm_importance_plot(
        shap_values=global_shap_matrix,
        feature_values=global_feat_matrix,
        feature_names=[f"global_{x}" for x in global_feature_names],
        save_path=os.path.join(save_dir, "global_beeswarm.svg"),
        top_k=top_k_global,
        title="Global feature importance and beeswarm distribution",
        max_points_per_feature=280,
    )

    region_flat_feature_names = _build_region_flat_feature_names(region_names, region_feature_names)
    region_summary_rows = save_beeswarm_importance_plot(
        shap_values=region_shap_matrix,
        feature_values=region_feat_matrix,
        feature_names=region_flat_feature_names,
        save_path=os.path.join(save_dir, "region_beeswarm.svg"),
        top_k=top_k_region,
        title="Region-related feature importance and beeswarm distribution",
        max_points_per_feature=280,
    )

    csv_path = os.path.join(save_dir, "selected_test_samples.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset_index", "image_id", "true_score", "pred_score"])
        writer.writeheader()
        for row in selected_meta:
            writer.writerow(row)

    global_summary_csv_path = os.path.join(save_dir, "global_beeswarm_summary.csv")
    with open(global_summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_name", "feature_display_name", "mean_abs_shap", "percentage"])
        writer.writeheader()
        for row in global_summary_rows:
            writer.writerow(row)

    region_summary_csv_path = os.path.join(save_dir, "region_beeswarm_summary.csv")
    with open(region_summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_name", "feature_display_name", "mean_abs_shap", "percentage"])
        writer.writeheader()
        for row in region_summary_rows:
            writer.writerow(row)

    return {
        "num_selected": len(selected_indices),
        "shap_method": shap_method,
        "selected_samples": selected_meta,
        "global_summary_rows": global_summary_rows,
        "region_summary_rows": region_summary_rows,
        "global_feature_names": list(global_feature_names),
        "region_flat_feature_names": region_flat_feature_names,
        "global_beeswarm_path": os.path.join(save_dir, "global_beeswarm.svg"),
        "region_beeswarm_path": os.path.join(save_dir, "region_beeswarm.svg"),
        "selected_csv_path": csv_path,
        "global_summary_csv_path": global_summary_csv_path,
        "region_summary_csv_path": region_summary_csv_path,
    }


# =========================================================
# Main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Batch SHAP beeswarm generator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/autodl-tmp/pythonProject2/Shap_analysis/outcomes",
        help="Output directory",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to explain")
    parser.add_argument("--sample-size", type=int, default=1000, help="Stratified sample size")
    parser.add_argument("--num-bins", type=int, default=10, help="Number of score bins for stratified sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--background-cache", type=str, default="", help="Optional background cache path")
    parser.add_argument("--background-size", type=int, default=64, help="Background sample count")
    parser.add_argument("--top-k-global", type=int, default=8, help="Top global features to display")
    parser.add_argument("--top-k-region", type=int, default=8, help="Top region-related features to display")
    parser.add_argument("--target-score", type=str, default="final", choices=["final", "global"], help="Target score for SHAP")
    parser.add_argument("--use-online-seg", action="store_true", help="Use online segmentation instead of offline masks")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N samples")
    args = parser.parse_args()

    if AestheticFeatureDataset is None:
        raise RuntimeError("AestheticFeatureDataset is unavailable. Please ensure utils.data_loader can be imported.")

    _ensure_dir(args.output_dir)
    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("Batch SHAP beeswarm generation")
    print("=" * 100)
    print(f"checkpoint: {args.checkpoint}")
    print(f"output_dir: {args.output_dir}")
    print(f"split: {args.split}")
    print(f"sample_size: {args.sample_size}")
    print(f"num_bins: {args.num_bins}")
    print(f"seed: {args.seed}")
    print(f"device: {device}")
    print(f"target_score: {args.target_score}")
    print(f"use_online_seg: {args.use_online_seg}")

    dataset = AestheticFeatureDataset(split=args.split, run_tag="default", strict_scaler=True)
    labels, image_ids = extract_labels_and_ids(dataset)
    selected_indices, bin_ids = proportional_stratified_sample_indices(
        labels=labels,
        sample_size=args.sample_size,
        num_bins=args.num_bins,
        seed=args.seed,
    )

    model = build_model(args.checkpoint, device)
    bg_global, bg_region = build_background(
        args.background_cache if args.background_cache else None,
        max_samples=args.background_size,
    )

    region_names = resolve_region_names()
    global_feature_names = resolve_global_feature_names()
    region_feature_names = resolve_region_feature_names()

    engine = BatchShapEngine(
        region_names=region_names,
        global_feature_names=global_feature_names,
        region_feature_names=region_feature_names,
        manual_forward_fn=manual_forward_from_intermediate_features,
        target_score=args.target_score,
    )

    result = explain_batch_and_plot(
        dataset=dataset,
        selected_indices=selected_indices,
        model=model,
        bg_global=bg_global,
        bg_region=bg_region,
        engine=engine,
        device=device,
        global_feature_names=global_feature_names,
        region_names=region_names,
        region_feature_names=region_feature_names,
        save_dir=args.output_dir,
        use_online_seg=args.use_online_seg,
        progress_every=args.progress_every,
        top_k_global=args.top_k_global,
        top_k_region=args.top_k_region,
    )

    bin_edges = np.linspace(1.0, 10.0, args.num_bins + 1, dtype=np.float32)
    selected_scores = labels[selected_indices]
    selected_bin_ids = np.digitize(selected_scores, bin_edges[1:-1], right=False)
    bin_summary: List[Dict[str, Any]] = []
    for b in range(args.num_bins):
        full_count = int(np.sum(bin_ids == b))
        sample_count = int(np.sum(selected_bin_ids == b))
        bin_summary.append({
            "bin_index": b,
            "score_range": [float(bin_edges[b]), float(bin_edges[b + 1])],
            "full_count": full_count,
            "sample_count": sample_count,
        })

    json_path = os.path.join(args.output_dir, "batch_beeswarm_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "split": args.split,
                "sample_size": int(args.sample_size),
                "seed": int(args.seed),
                "num_bins": int(args.num_bins),
                "background_size": int(args.background_size),
                "target_score": args.target_score,
                "use_online_seg": bool(args.use_online_seg),
                "dataset_size": int(len(dataset)),
                "selected_indices": selected_indices.tolist(),
                "image_ids_for_selected": [image_ids[int(i)] for i in selected_indices],
                "score_bin_summary": bin_summary,
                **result,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 100)
    print("Done")
    print("=" * 100)
    print(f"Selected {len(selected_indices)} samples from split={args.split}")
    print(f"SHAP method: {result['shap_method']}")
    print(f"Global beeswarm figure: {result['global_beeswarm_path']}")
    print(f"Region beeswarm figure: {result['region_beeswarm_path']}")
    print(f"Selected sample CSV: {result['selected_csv_path']}")
    print(f"Global summary CSV: {result['global_summary_csv_path']}")
    print(f"Region summary CSV: {result['region_summary_csv_path']}")
    print(f"Summary JSON: {json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
