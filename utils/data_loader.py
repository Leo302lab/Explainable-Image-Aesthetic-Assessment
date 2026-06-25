import os
import random
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

import utils.config as cfg


EXCLUDE_COLS = {
    cfg.ID_COLUMN,
    "std_score",
    "mean_score",
    "label",
    cfg.LABEL_COLUMN,
    cfg.LABEL_RAW_COLUMN,
    "split",
    "set",
    # AVA 评分分布列（明确排除，避免泄漏）
    "score_1", "score_2", "score_3", "score_4", "score_5",
    "score_6", "score_7", "score_8", "score_9", "score_10",
    "dist_1", "dist_2", "dist_3", "dist_4", "dist_5",
    "dist_6", "dist_7", "dist_8", "dist_9", "dist_10",
    "rating_1", "rating_2", "rating_3", "rating_4", "rating_5",
    "rating_6", "rating_7", "rating_8", "rating_9", "rating_10",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
}


def _semantic_region_names() -> List[str]:
    if isinstance(cfg.SEMANTIC_LABELS, dict):
        if all(isinstance(k, int) for k in cfg.SEMANTIC_LABELS.keys()):
            return [str(cfg.SEMANTIC_LABELS[i]).lower() for i in sorted(cfg.SEMANTIC_LABELS.keys())]
        return [str(k).lower() for k in cfg.SEMANTIC_LABELS.keys()]
    return [str(x).lower() for x in cfg.SEMANTIC_LABELS]


def _expected_core_feature_cols() -> List[str]:
    globals_ = [f"global_{k}" for k in cfg.CORE_FEATURE_ORDER]
    region_prefixes = _semantic_region_names()
    regions_ = [f"{p}_{k}" for p in region_prefixes for k in cfg.CORE_FEATURE_ORDER]
    return globals_ + regions_


def _expected_region_meta_cols() -> List[str]:
    region_prefixes = _semantic_region_names()
    return [f"{p}_{k}" for p in region_prefixes for k in cfg.REGION_META_FEATS]


def _normalize_image_id_string(x) -> str:
    s = str(x).strip()
    if s.lower().endswith((".jpg", ".png", ".jpeg")):
        s = os.path.splitext(s)[0]
    try:
        if s.endswith(".0"):
            fv = float(s)
            if fv.is_integer():
                s = str(int(fv))
    except Exception:
        pass
    return s


def _candidate_image_ids(img_id: str) -> List[str]:
    s = _normalize_image_id_string(img_id)
    candidates = [s]
    if s.isdigit():
        candidates.append(str(int(s)))
        candidates.append(s.zfill(5))
        candidates.append(s.zfill(6))

    out = []
    seen = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _resolve_image_path(image_dir: str, img_id: str) -> Optional[str]:
    exts = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]
    for cid in _candidate_image_ids(img_id):
        for ext in exts:
            p = os.path.join(image_dir, f"{cid}{ext}")
            if os.path.exists(p):
                return p
    return None


def _resolve_mask_path(mask_dir: str, img_id: str) -> Optional[str]:
    exts = [".png", ".PNG"]
    for cid in _candidate_image_ids(img_id):
        for ext in exts:
            p = os.path.join(mask_dir, f"{cid}{ext}")
            if os.path.exists(p):
                return p
    return None


def _load_mask_preserve_labels(mask_path: str) -> Image.Image:
    mask = Image.open(mask_path)
    if mask.mode in ("P", "L", "I"):
        return mask
    raise ValueError(
        f"Unsupported mask mode: {mask.mode}. "
        "Please provide indexed semantic masks instead of RGB masks."
    )


def _to_hw(size_like) -> Tuple[int, int]:
    if isinstance(size_like, int):
        return size_like, size_like
    if isinstance(size_like, (tuple, list)) and len(size_like) >= 2:
        return int(size_like[0]), int(size_like[1])
    return 224, 224


class AestheticFeatureDataset(Dataset):
    """
    离线 mask 主链版：
    1. ResNet / ViT / SegFormer 三路输入分开返回；
    2. 返回 semantic_mask，供 full_model 直接做区域提取；
    3. core 特征标准化时，对无效区域采用 valid-aware 统计；
    4. 归一化后无效区域特征强制置0；
    5. meta 单独返回，不并入主特征；
    6. 只使用单点评分 label / label_raw，不使用 AVA 10 维评分分布；
    7. 训练阶段只启用“离线特征安全”的轻量增强。
    8. 主特征只使用 108 维核心特征：12 维 global + 8*12 维 region。
    """

    def __init__(
        self,
        split: str = "train",
        feature_dir: str = None,
        run_tag: str = "default",
        strict_scaler: bool = True,
    ):
        self.split = str(split)
        self.run_tag = str(run_tag)
        self.strict_scaler = bool(strict_scaler)
        self.is_train = (self.split == "train")

        self.feature_dir = feature_dir or cfg.FEATURE_DIR
        self.image_dir = cfg.IMAGE_DIR
        self.semantic_mask_dir = str(
            getattr(cfg, "SEMANTIC_MASK_DIR", "/root/autodl-tmp/Data_preprocess/semantic_masks_finetuned")
        )

        self.region_names = _semantic_region_names()
        self.region_count = len(self.region_names)

        self.global_core_dim = int(
            getattr(cfg, "GLOBAL_CORE_DIM", getattr(cfg, "GLOBAL_HANDCRAFTED_DIM", len(cfg.CORE_FEATURE_ORDER)))
        )
        self.region_core_dim = int(getattr(cfg, "REGION_CORE_DIM", len(cfg.CORE_FEATURE_ORDER)))
        self.region_meta_dim = int(getattr(cfg, "REGION_META_PER_REGION", len(cfg.REGION_META_FEATS)))

        self.resnet_hw = _to_hw(cfg.IMAGE_SIZE)
        self.vit_hw = _to_hw(getattr(cfg, "IMAGE_SIZE", 224))
        self.seg_hw = _to_hw(getattr(cfg, "SEG_IMAGE_SIZE", cfg.IMAGE_SIZE))
        self.mask_hw = _to_hw(cfg.IMAGE_SIZE)

        self.mean = list(cfg.MEAN)
        self.std = list(cfg.STD)
        self.vit_mean = list(getattr(cfg, "VIT_MEAN", self.mean))
        self.vit_std = list(getattr(cfg, "VIT_STD", self.std))
        self.seg_mean = list(getattr(cfg, "SEG_MEAN", self.mean))
        self.seg_std = list(getattr(cfg, "SEG_STD", self.std))

        self.use_train_aug = bool(getattr(cfg, "USE_TRAIN_AUG", True))
        self.aug_enable_hflip = bool(getattr(cfg, "AUG_ENABLE_HFLIP", False))
        self.aug_hflip_prob = float(getattr(cfg, "AUG_HFLIP_PROB", 0.5))
        self.aug_color_jitter_prob = float(getattr(cfg, "AUG_COLOR_JITTER_PROB", 0.8))
        self.aug_brightness = float(getattr(cfg, "AUG_BRIGHTNESS", 0.12))
        self.aug_contrast = float(getattr(cfg, "AUG_CONTRAST", 0.12))
        self.aug_saturation = float(getattr(cfg, "AUG_SATURATION", 0.10))
        self.aug_hue = float(getattr(cfg, "AUG_HUE", 0.02))
        self.aug_blur_prob = float(getattr(cfg, "AUG_GAUSSIAN_BLUR_PROB", 0.15))
        blur_radius = getattr(cfg, "AUG_GAUSSIAN_BLUR_RADIUS", (0.1, 1.2))
        if isinstance(blur_radius, (tuple, list)) and len(blur_radius) == 2:
            self.aug_blur_radius = (float(blur_radius[0]), float(blur_radius[1]))
        else:
            self.aug_blur_radius = (0.1, 1.2)
        self.aug_grayscale_prob = float(getattr(cfg, "AUG_GRAYSCALE_PROB", 0.05))

        self._missing_mask_warned = set()

        feature_path = os.path.join(self.feature_dir, f"{self.split}_features.csv")
        scaler_path = os.path.join(self.feature_dir, f"feature_scaler_{self.run_tag}.pkl")

        if not os.path.exists(feature_path):
            alt_path = os.path.join(self.feature_dir, "dataset.csv")
            if os.path.exists(alt_path):
                print(f"⚠️ 未找到 {feature_path}，尝试加载总表 {alt_path}")
                feature_path = alt_path
            else:
                raise FileNotFoundError(f"❌ 找不到特征文件: {feature_path}")

        df = pd.read_csv(feature_path)
        if "split" in df.columns:
            df = df[df["split"] == self.split].reset_index(drop=True)

        if cfg.ID_COLUMN not in df.columns:
            raise KeyError(f"❌ CSV中找不到 ID 列: {cfg.ID_COLUMN}")

        self.image_ids = df[cfg.ID_COLUMN].apply(_normalize_image_id_string).values

        if cfg.LABEL_COLUMN in df.columns:
            self.labels = pd.to_numeric(df[cfg.LABEL_COLUMN], errors="coerce").values.astype(np.float32)
        elif "mean_score" in df.columns:
            print(f"⚠️ [{self.split}] 未找到 '{cfg.LABEL_COLUMN}'，使用 'mean_score' 并归一化到 [0,1]")
            mean_score = pd.to_numeric(df["mean_score"], errors="coerce").values.astype(np.float32)
            self.labels = (mean_score - 1.0) / 9.0
        else:
            raise KeyError(f"❌ 找不到训练标签列 ({cfg.LABEL_COLUMN} 或 mean_score)")

        self.has_label_raw = False
        self.labels_raw = None
        if cfg.LABEL_RAW_COLUMN in df.columns:
            self.labels_raw = pd.to_numeric(df[cfg.LABEL_RAW_COLUMN], errors="coerce").values.astype(np.float32)
            self.has_label_raw = True
        elif "mean_score" in df.columns:
            self.labels_raw = pd.to_numeric(df["mean_score"], errors="coerce").values.astype(np.float32)
            self.has_label_raw = True
        else:
            if cfg.LABEL_IS_NORMALIZED_0_1:
                self.labels_raw = self.labels * 9.0 + 1.0
                self.has_label_raw = True
            else:
                print(f"⚠️ [{self.split}] 无法生成 Label Raw (1~10)，测试指标可能不准确。")

        # 明确关闭 10 维评分分布
        self.has_score_distribution = False
        self.score_distribution = None

        expected_core_cols = _expected_core_feature_cols()
        expected_meta_cols = _expected_region_meta_cols()

        missing_core = [c for c in expected_core_cols if c not in df.columns]
        missing_meta = [c for c in expected_meta_cols if c not in df.columns]

        if len(missing_core) == 0:
            self.feature_cols = expected_core_cols
        else:
            raw_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c not in expected_meta_cols]
            self.feature_cols = raw_cols[:cfg.MANUAL_FEATURE_DIM] if cfg.MANUAL_FEATURE_DIM else raw_cols
            print(f"⚠️ [Dataset:{self.split}] 期望核心特征列缺失，回退为原始列顺序。缺失示例: {missing_core[:5]}")
            print(f"   >>> 请确认前 {self.global_core_dim} 列确实是 global_ 特征 <<<")

        raw_features = df[self.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)

        expected_total_core_dim = self.global_core_dim + self.region_count * self.region_core_dim
        if raw_features.shape[1] != expected_total_core_dim:
            raise ValueError(
                f"Core feature dim mismatch: got={raw_features.shape[1]}, "
                f"expected={expected_total_core_dim} "
                f"(global={self.global_core_dim}, regions={self.region_count}x{self.region_core_dim})."
            )

        first_global_cols = [c.lower() for c in self.feature_cols[:self.global_core_dim]]
        if not all(x.startswith("global_") for x in first_global_cols):
            raise ValueError(f"前{self.global_core_dim}维不是global特征：{self.feature_cols[:self.global_core_dim]}")

        if len(missing_meta) == 0:
            meta_vals = df[expected_meta_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
            meta_vals = meta_vals.reshape(len(df), self.region_count, len(cfg.REGION_META_FEATS))
            self.region_valid = (meta_vals[..., 0] > 0.5)
            self.region_area_ratio = np.clip(meta_vals[..., 1], 0.0, 1.0).astype(np.float32)
        else:
            print(f"⚠️ [Dataset:{self.split}] 缺失 region meta 字段，回退为由核心特征推断 valid。缺失示例: {missing_meta[:4]}")
            region_part = raw_features[:, self.global_core_dim:]
            hand_region_dim = (raw_features.shape[1] - self.global_core_dim) // self.region_count
            region_part = region_part.reshape(len(df), self.region_count, hand_region_dim)
            self.region_valid = (np.abs(region_part).sum(axis=-1) > 1e-8)
            self.region_area_ratio = np.zeros((len(df), self.region_count), dtype=np.float32)

        self.features = self._fit_transform_core_features(raw_features=raw_features, scaler_path=scaler_path)

        print(
            f"🚀 [{self.split:<5}] Loaded: {len(self.features)} items | "
            f"Core Feat Dim: {self.features.shape[1]} | Meta: valid+area_ratio | "
            f"Mask Dir: {self.semantic_mask_dir}"
        )

    def _fit_transform_core_features(self, raw_features: np.ndarray, scaler_path: str) -> np.ndarray:
        n = raw_features.shape[0]
        global_part = raw_features[:, :self.global_core_dim]
        region_feature_dim = raw_features.shape[1] - self.global_core_dim

        self.region_core_dim = region_feature_dim // self.region_count
        region_part = raw_features[:, self.global_core_dim:].reshape(n, self.region_count, self.region_core_dim)
        valid_mask = self.region_valid.astype(bool)

        if self.is_train:
            global_means = np.mean(global_part, axis=0).astype(np.float32)
            global_stds = (np.std(global_part, axis=0) + 1e-6).astype(np.float32)

            region_means = np.zeros((self.region_count, self.region_core_dim), dtype=np.float32)
            region_stds = np.ones((self.region_count, self.region_core_dim), dtype=np.float32)
            for r in range(self.region_count):
                valid_r = valid_mask[:, r]
                if valid_r.any():
                    vals = region_part[valid_r, r, :]
                    region_means[r] = vals.mean(axis=0).astype(np.float32)
                    region_stds[r] = (vals.std(axis=0) + 1e-6).astype(np.float32)
                else:
                    region_means[r] = 0.0
                    region_stds[r] = 1.0

            joblib.dump(
                {
                    "global_means": global_means,
                    "global_stds": global_stds,
                    "region_means": region_means,
                    "region_stds": region_stds,
                    "feature_cols": self.feature_cols,
                },
                scaler_path,
            )
        else:
            if os.path.exists(scaler_path):
                stats = joblib.load(scaler_path)
                if "global_means" in stats:
                    global_means = stats["global_means"]
                    global_stds = stats["global_stds"]
                    region_means = stats["region_means"]
                    region_stds = stats["region_stds"]
                else:
                    print(f"⚠️ [{self.split}] 读取到旧版 scaler，将退化为旧标准化逻辑。建议重新生成 train scaler。")
                    means = stats["means"]
                    stds = stats["stds"]
                    normalized_feats = (raw_features - means) / stds
                    normalized_feats = np.clip(normalized_feats, -5.0, 5.0).astype(np.float32)
                    region_norm = normalized_feats[:, self.global_core_dim:].reshape(n, self.region_count, self.region_core_dim)
                    region_norm[~valid_mask] = 0.0
                    normalized_feats[:, self.global_core_dim:] = region_norm.reshape(n, -1)
                    return normalized_feats
            else:
                msg = f"❌ [{self.split}] 缺失 Scaler: {scaler_path}。请先运行 Train 生成。"
                if self.strict_scaler:
                    raise FileNotFoundError(msg)
                print(f"⚠️ {msg} (将使用当前集统计，存在泄露风险)")

                global_means = np.mean(global_part, axis=0).astype(np.float32)
                global_stds = (np.std(global_part, axis=0) + 1e-6).astype(np.float32)
                region_means = np.zeros((self.region_count, self.region_core_dim), dtype=np.float32)
                region_stds = np.ones((self.region_count, self.region_core_dim), dtype=np.float32)
                for r in range(self.region_count):
                    valid_r = valid_mask[:, r]
                    if valid_r.any():
                        vals = region_part[valid_r, r, :]
                        region_means[r] = vals.mean(axis=0).astype(np.float32)
                        region_stds[r] = (vals.std(axis=0) + 1e-6).astype(np.float32)
                    else:
                        region_means[r] = 0.0
                        region_stds[r] = 1.0

        norm_global = (global_part - global_means[None, :]) / global_stds[None, :]
        norm_region = (region_part - region_means[None, :, :]) / region_stds[None, :, :]
        norm_region[~valid_mask] = 0.0

        normalized_feats = np.concatenate([norm_global, norm_region.reshape(n, -1)], axis=1)
        normalized_feats = np.clip(normalized_feats, -5.0, 5.0).astype(np.float32)
        return normalized_feats

    def _apply_train_safe_augment(self, img: Image.Image, mask: Optional[Image.Image]):
        if (not self.is_train) or (not self.use_train_aug):
            return img, mask

        if self.aug_enable_hflip and random.random() < self.aug_hflip_prob:
            img = TF.hflip(img)
            if mask is not None:
                mask = TF.hflip(mask)

        if random.random() < self.aug_color_jitter_prob:
            if self.aug_brightness > 0:
                img = TF.adjust_brightness(img, 1.0 + random.uniform(-self.aug_brightness, self.aug_brightness))
            if self.aug_contrast > 0:
                img = TF.adjust_contrast(img, 1.0 + random.uniform(-self.aug_contrast, self.aug_contrast))
            if self.aug_saturation > 0:
                img = TF.adjust_saturation(img, 1.0 + random.uniform(-self.aug_saturation, self.aug_saturation))
            if self.aug_hue > 0:
                img = TF.adjust_hue(img, random.uniform(-self.aug_hue, self.aug_hue))

        if self.aug_grayscale_prob > 0 and random.random() < self.aug_grayscale_prob:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)

        if self.aug_blur_prob > 0 and random.random() < self.aug_blur_prob:
            radius = random.uniform(self.aug_blur_radius[0], self.aug_blur_radius[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        return img, mask

    def _prepare_multi_inputs_and_mask(self, raw_image: Image.Image, raw_mask: Optional[Image.Image]):
        img, mask = self._apply_train_safe_augment(raw_image, raw_mask)

        img_resnet = TF.resize(img, self.resnet_hw, interpolation=InterpolationMode.BILINEAR)
        img_vit = TF.resize(img, self.vit_hw, interpolation=InterpolationMode.BILINEAR)
        img_seg = TF.resize(img, self.seg_hw, interpolation=InterpolationMode.BILINEAR)

        image_tensor = TF.normalize(TF.to_tensor(img_resnet), mean=self.mean, std=self.std)
        vit_tensor = TF.normalize(TF.to_tensor(img_vit), mean=self.vit_mean, std=self.vit_std)
        seg_tensor = TF.normalize(TF.to_tensor(img_seg), mean=self.seg_mean, std=self.seg_std)

        if mask is not None:
            mask_resized = TF.resize(mask, self.mask_hw, interpolation=InterpolationMode.NEAREST)
            mask_np = np.array(mask_resized, dtype=np.int64)
            uniq = np.unique(mask_np)
            if uniq.size > 0 and (uniq.min() < 0 or uniq.max() >= self.region_count):
                raise ValueError(
                    f"Mask label out of range: unique={uniq.tolist()}, expected in [0, {self.region_count - 1}]"
                )
            semantic_mask = torch.from_numpy(mask_np).long()
        else:
            semantic_mask = torch.zeros(self.mask_hw[0], self.mask_hw[1], dtype=torch.long)

        return image_tensor, vit_tensor, seg_tensor, semantic_mask

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        found_path = _resolve_image_path(self.image_dir, img_id)
        mask_path = _resolve_mask_path(self.semantic_mask_dir, img_id)

        try:
            if found_path is None:
                raise FileNotFoundError(f"未找到图像文件，image_id={img_id}")

            raw_image = Image.open(found_path).convert("RGB")
            raw_mask = _load_mask_preserve_labels(mask_path) if mask_path is not None else None
            image_tensor, vit_tensor, seg_tensor, semantic_mask = self._prepare_multi_inputs_and_mask(raw_image, raw_mask)

            if mask_path is None and img_id not in self._missing_mask_warned:
                print(f"⚠️ 缺失离线mask: image_id={img_id} | dir={self.semantic_mask_dir}，将使用全0 mask")
                self._missing_mask_warned.add(img_id)
        except Exception as e:
            rh, rw = self.resnet_hw
            vh, vw = self.vit_hw
            sh, sw = self.seg_hw
            mh, mw = self.mask_hw
            print(f"⚠️ 读取图像/掩码失败: image_id={img_id} | path={found_path} | mask={mask_path} | error: {e}")
            image_tensor = torch.zeros(3, rh, rw, dtype=torch.float32)
            vit_tensor = torch.zeros(3, vh, vw, dtype=torch.float32)
            seg_tensor = torch.zeros(3, sh, sw, dtype=torch.float32)
            semantic_mask = torch.zeros(mh, mw, dtype=torch.long)

        sample = {
            "image": image_tensor,
            "vit_image": vit_tensor,
            "seg_image": seg_tensor,
            "semantic_mask": semantic_mask,
            "features": torch.from_numpy(self.features[idx]),
            "label": torch.tensor([self.labels[idx]], dtype=torch.float32),
            "region_valid": torch.from_numpy(self.region_valid[idx].astype(np.float32)),
            "region_area_ratio": torch.from_numpy(self.region_area_ratio[idx].astype(np.float32)),
            "image_id": str(img_id),
        }

        if self.has_label_raw:
            sample["label_raw"] = torch.tensor([self.labels_raw[idx]], dtype=torch.float32)

        return sample


def get_feature_dataloader(
    split="train",
    batch_size=4,
    shuffle=None,
    num_workers=0,
    pin_memory=False,
    worker_init_fn=None,
    generator=None,
    feature_dir=None,
    run_tag="default",
    strict_scaler=True,
):
    dataset = AestheticFeatureDataset(
        split=split,
        feature_dir=feature_dir,
        run_tag=run_tag,
        strict_scaler=strict_scaler,
    )

    if shuffle is None:
        shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=(num_workers > 0),
        drop_last=(split == "train"),
    )
