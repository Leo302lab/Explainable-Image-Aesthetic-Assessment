from pathlib import Path
import warnings
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim

try:
    # pip install brisque libsvm-official
    from brisque import BRISQUE
    _HAS_BRISQUE = True
except Exception:
    BRISQUE = None
    _HAS_BRISQUE = False

print("PYTHON =", sys.executable)
print("_HAS_BRISQUE =", _HAS_BRISQUE)

# ================= 1. 路径配置 =================
BASE_DIR = Path("/root/autodl-tmp")
CLEANED_IMAGE_DIR = BASE_DIR / "Data_preprocess" / "cleaned"
SEMANTIC_MASK_DIR = BASE_DIR / "Data_preprocess" / "semantic_masks"
FEATURE_DIR = BASE_DIR / "Data_preprocess" / "features_Fixed_final_design_roi_hardmask"
RAW_DATA_DIR = BASE_DIR / "AVA＿dataset"
SPLIT_DIR = RAW_DATA_DIR / "splits"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# 语义标签顺序必须与训练一致
SEMANTIC_LABELS = {
    0: "background",
    1: "face",
    2: "sky",
    3: "building",
    4: "vegetation",
    5: "water",
    6: "object",
    7: "noise",
}
REGION_ORDER = [SEMANTIC_LABELS[i] for i in range(len(SEMANTIC_LABELS))]
NUM_REGIONS = len(REGION_ORDER)

# 原始设计对齐的 12 维核心特征
CORE_FEATURE_ORDER = [
    "edge_sharpness",
    "nr_sharpness",
    "color_cast",
    "saturation",
    "color_balance",
    "over_exposure",
    "under_exposure",
    "brightness_uniformity",
    "composition",
    "symmetry",
    "texture",
    "contrast",
]

REGION_META_ORDER = [
    "valid",
    "area_ratio",
]


class DesignAlignedFeatureExtractorROIHardMask:
    """
    ROI-align 优先 + hard mask 唯一输入的严格对齐版本：
    1) global / region 都提 12 维核心特征
    2) 所有特征统一到 [0,1]，且“值越高越好”
    3) 不再使用 soft-mask；只读取 .png 硬 mask
    4) region 特征先按 hard mask 提 ROI，再在 ROI 内计算特征
    5) region nr_sharpness 基于 ROI patch 做 BRISQUE；异常时回退代理值
    6) brightness_uniformity 使用亮度方差取反，更贴近原始设计表述
    7) 只保留单点评分 label / label_raw，不再写 10 维评分分布
    8) 缺失评分标签时直接跳过，不再默认写入 5.5
    """

    def __init__(
        self,
        min_region_pixels=20,
        use_true_brisque_for_global=True,
        use_true_brisque_for_regions=True,
        brisque_min_patch_side=32,
        brisque_max_patch_side=256,
        glcm_levels=16,
        roi_output_size=(128, 128),
        roi_margin_ratio=0.05,
        composition_subject_region_names=("face", "object", "building"),
        composition_fallback_exclude=("background", "noise"),
    ):
        self.min_region_pixels = int(min_region_pixels)
        self.use_true_brisque_for_global = bool(use_true_brisque_for_global)
        self.use_true_brisque_for_regions = bool(use_true_brisque_for_regions)
        self.brisque_min_patch_side = int(brisque_min_patch_side)
        self.brisque_max_patch_side = int(brisque_max_patch_side)
        self.glcm_levels = int(glcm_levels)
        self.roi_output_size = tuple(roi_output_size)
        self.roi_margin_ratio = float(roi_margin_ratio)

        self.composition_subject_region_names = tuple(str(x) for x in composition_subject_region_names)
        self.composition_fallback_exclude = tuple(str(x) for x in composition_fallback_exclude)

        self.brisque_model = BRISQUE() if _HAS_BRISQUE else None
        self._warned_brisque = False

        self.region_name_to_idx = {name: idx for idx, name in SEMANTIC_LABELS.items()}

    # ================= 2. 读取标签（只保留单点评分） =================
    def load_ava_scores(self):
        """
        返回:
            score_mapping[image_id] = {
                "label": [0,1] 归一化分数,
                "label_raw": [1,10] 原始均值分,
            }

        读取策略：
        1) 优先按 AVA 标准 txt 格式解析：
           [index, image_id, rating_1, ..., rating_10, ...]
        2) 若不满足，再退回到旧格式：
           [image_id, rating_1, ..., rating_10, ...]
        """
        print("🔍 加载 AVA 评分数据...")
        possible_files = [
            RAW_DATA_DIR / "AVA.txt",
            RAW_DATA_DIR / "cleaned_ava.csv",
        ]

        def _is_number(x):
            try:
                float(x)
                return True
            except Exception:
                return False

        score_mapping = {}
        for file_path in possible_files:
            if not file_path.exists():
                continue

            try:
                if file_path.suffix.lower() == ".txt":
                    df = pd.read_csv(file_path, sep=r"\s+", header=None)

                    for _, row in df.iterrows():
                        vals = row.values.tolist()

                        image_id = None
                        ratings = None

                        # 标准 AVA 格式：
                        # [index, image_id, rating_1 ... rating_10, ...]
                        if (
                            len(vals) >= 12
                            and _is_number(vals[0])
                            and _is_number(vals[1])
                            and all(_is_number(v) for v in vals[2:12])
                        ):
                            image_id = str(vals[1]).strip()
                            ratings = np.asarray(vals[2:12], dtype=np.float32)

                        # 兼容旧格式：
                        # [image_id, rating_1 ... rating_10, ...]
                        elif (
                            len(vals) >= 11
                            and all(_is_number(v) for v in vals[1:11])
                        ):
                            image_id = str(vals[0]).strip()
                            ratings = np.asarray(vals[1:11], dtype=np.float32)

                        if image_id is None or ratings is None:
                            continue

                        total = float(ratings.sum())
                        if total <= 0:
                            continue

                        mean_s = float(np.sum(np.arange(1, 11, dtype=np.float32) * ratings) / total)
                        score_mapping[self._normalize_image_id(image_id)] = {
                            "label": (mean_s - 1.0) / 9.0,
                            "label_raw": mean_s,
                        }

                elif file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path)
                    for _, row in df.iterrows():
                        img_id = self._normalize_image_id(row["image_id"])
                        mean_s = float(row["mean_score"])
                        score_mapping[img_id] = {
                            "label": (mean_s - 1.0) / 9.0,
                            "label_raw": mean_s,
                        }

                if score_mapping:
                    print(f"✅ 成功加载 {len(score_mapping)} 条评分记录 (来自 {file_path.name})")
                    return score_mapping

            except Exception as e:
                print(f"❌ 读取错误: {e}")

        return {}

    # ================= 3. 基础工具 =================
    def _normalize_image_id(self, image_id):
        s = str(image_id).strip()
        s = Path(s).stem
        if s.isdigit():
            return str(int(s)).zfill(5)
        return s

    def _invalid_core_features(self, prefix):
        return {f"{prefix}_{name}": 0.0 for name in CORE_FEATURE_ORDER}

    def _region_meta(self, region_name, region_mass, total_pixels, is_valid):
        area_ratio = float(region_mass) / float(total_pixels) if total_pixels > 0 else 0.0
        area_ratio = float(np.clip(area_ratio, 0.0, 1.0))
        return {
            f"{region_name}_valid": float(is_valid),
            f"{region_name}_area_ratio": area_ratio,
        }

    @staticmethod
    def _safe_clip01(x):
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _weighted_mean(values, weights):
        denom = float(np.sum(weights))
        if denom <= 1e-8:
            return 0.0
        return float(np.sum(values * weights) / denom)

    @staticmethod
    def _weighted_var(values, weights):
        denom = float(np.sum(weights))
        if denom <= 1e-8:
            return 0.0
        mean = np.sum(values * weights) / denom
        var = np.sum(weights * ((values - mean) ** 2)) / denom
        return float(max(var, 0.0))

    def _resolve_image_path(self, image_id):
        img_id_str = self._normalize_image_id(image_id)
        raw_candidates = [img_id_str]
        if img_id_str.isdigit():
            raw_candidates.append(str(int(img_id_str)))

        for stem in raw_candidates:
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                p = CLEANED_IMAGE_DIR / f"{stem}{ext}"
                if p.exists():
                    return p
        return None

    def _load_hard_mask(self, image_id, h, w):
        """
        只读取硬 mask，不再读取 soft-mask。
        约定：
          {image_id}.png 为单通道语义类别图，像素值 ∈ [0, NUM_REGIONS-1]
        返回:
          mask: [H, W] uint8
          mask_type: "hard" / "none"
        """
        img_id_str = self._normalize_image_id(image_id)
        png_path = SEMANTIC_MASK_DIR / f"{img_id_str}.png"

        if not png_path.exists():
            return None, "none"

        mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, "none"

        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        return mask.astype(np.uint8), "hard"

    def _prepare_images(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        img_hsv = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[..., 0] = img_hsv[..., 0] / 180.0
        img_hsv[..., 1] = img_hsv[..., 1] / 255.0
        img_hsv[..., 2] = img_hsv[..., 2] / 255.0

        img_lab = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        return img_rgb, img_gray, img_hsv, img_lab

    def _compute_gradient_maps(self, img_gray):
        lap = np.abs(cv2.Laplacian(img_gray, cv2.CV_32F))
        sobel_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return lap, grad_mag

    def _local_std_map(self, img_gray, ksize=9):
        mean = cv2.boxFilter(img_gray, ddepth=-1, ksize=(ksize, ksize), normalize=True)
        mean_sq = cv2.boxFilter(img_gray ** 2, ddepth=-1, ksize=(ksize, ksize), normalize=True)
        var = np.maximum(mean_sq - mean ** 2, 0.0)
        return np.sqrt(var)

    # ================= 4. ROI 相关函数 =================
    def _binary_mask_to_box(self, binary_mask, margin_ratio=None, min_side=8):
        """
        从二值 mask 提取 ROI bbox，返回 (x1, y1, x2, y2)。
        """
        if margin_ratio is None:
            margin_ratio = self.roi_margin_ratio

        ys, xs = np.where(binary_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        h, w = binary_mask.shape
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1

        bw = x2 - x1
        bh = y2 - y1

        mx = max(1, int(round(bw * margin_ratio)))
        my = max(1, int(round(bh * margin_ratio)))

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        if (x2 - x1) < min_side:
            cx = (x1 + x2) // 2
            half = max(min_side // 2, 1)
            x1 = max(0, cx - half)
            x2 = min(w, cx + half)

        if (y2 - y1) < min_side:
            cy = (y1 + y2) // 2
            half = max(min_side // 2, 1)
            y1 = max(0, cy - half)
            y2 = min(h, cy + half)

        if x2 <= x1 or y2 <= y1:
            return None

        return int(x1), int(y1), int(x2), int(y2)

    def _roi_align_ndarray(self, arr, box, output_size=None, is_mask=False):
        """
        轻量 ROI align:
        - 图像/特征图: 双线性插值
        - mask: 最近邻插值
        支持 [H, W] 或 [H, W, C]
        """
        if output_size is None:
            output_size = self.roi_output_size

        x1, y1, x2, y2 = box
        crop = arr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        out_h, out_w = output_size
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        out = cv2.resize(crop, (out_w, out_h), interpolation=interp)
        return out

    def _extract_region_roi_payload(
        self,
        img_rgb,
        img_gray,
        img_hsv,
        img_lab,
        lap_map,
        grad_map,
        semantic_mask,
        class_id,
        roi_size=None,
        margin_ratio=None,
    ):
        """
        根据硬 mask 提取某一类区域的 ROI 对齐结果。
        返回:
          payload: dict or None
          region_mass: 原图该区域像素数
          is_valid: bool
        """
        if roi_size is None:
            roi_size = self.roi_output_size
        if margin_ratio is None:
            margin_ratio = self.roi_margin_ratio

        binary_mask = (semantic_mask == class_id).astype(np.uint8)
        region_mass = float(binary_mask.sum())

        if region_mass < float(self.min_region_pixels):
            return None, region_mass, False

        box = self._binary_mask_to_box(binary_mask, margin_ratio=margin_ratio, min_side=8)
        if box is None:
            return None, region_mass, False

        roi_rgb = self._roi_align_ndarray(img_rgb, box, roi_size, is_mask=False)
        roi_gray = self._roi_align_ndarray(img_gray, box, roi_size, is_mask=False)
        roi_hsv = self._roi_align_ndarray(img_hsv, box, roi_size, is_mask=False)
        roi_lab = self._roi_align_ndarray(img_lab, box, roi_size, is_mask=False)
        roi_lap = self._roi_align_ndarray(lap_map, box, roi_size, is_mask=False)
        roi_grad = self._roi_align_ndarray(grad_map, box, roi_size, is_mask=False)
        roi_mask = self._roi_align_ndarray(binary_mask.astype(np.float32), box, roi_size, is_mask=True)

        if any(x is None for x in [roi_rgb, roi_gray, roi_hsv, roi_lab, roi_lap, roi_grad, roi_mask]):
            return None, region_mass, False

        roi_weights = (roi_mask > 0.5).astype(np.float32)
        if float(roi_weights.sum()) < float(self.min_region_pixels):
            return None, region_mass, False

        payload = {
            "img_rgb": roi_rgb.astype(np.float32),
            "img_gray": roi_gray.astype(np.float32),
            "img_hsv": roi_hsv.astype(np.float32),
            "img_lab": roi_lab.astype(np.float32),
            "lap_map": roi_lap.astype(np.float32),
            "grad_map": roi_grad.astype(np.float32),
            "weights": roi_weights.astype(np.float32),
            "box": box,
        }
        return payload, region_mass, True

    # ================= 5. 特征辅助函数 =================
    def _resize_if_needed(self, img_u8, max_side=256):
        h, w = img_u8.shape[:2]
        max_hw = max(h, w)
        if max_hw <= max_side:
            return img_u8
        scale = max_side / float(max_hw)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        return cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_AREA)

    def _is_degenerate_patch(self, patch_u8):
        if patch_u8 is None or patch_u8.size == 0:
            return True
        if patch_u8.ndim != 3 or patch_u8.shape[2] != 3:
            return True
        h, w = patch_u8.shape[:2]
        if min(h, w) < self.brisque_min_patch_side:
            return True

        gray = cv2.cvtColor(patch_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if float(gray.std()) < 1.5:
            return True
        if float(np.mean(gray > 250)) > 0.98:
            return True
        if float(np.mean(gray < 5)) > 0.98:
            return True
        return False

    def _roi_masked_patch_for_brisque(self, img_rgb, weights):
        """
        在 ROI 对齐后的局部区域内，使用 mask 约束有效区域，
        并用区域均值填补外部，供 BRISQUE 评估。
        """
        if img_rgb is None or weights is None:
            return None
        if img_rgb.size == 0 or weights.size == 0:
            return None

        hard = (weights > 0.5).astype(np.float32)
        if hard.sum() < self.min_region_pixels:
            return None

        region_vals = img_rgb[hard > 0.5]
        if region_vals.size == 0:
            return None

        mean_color = region_vals.mean(axis=0, keepdims=True).reshape(1, 1, 3)
        composed = img_rgb * hard[..., None] + mean_color * (1.0 - hard[..., None])
        patch_u8 = np.clip(composed * 255.0, 0, 255).astype(np.uint8)
        return patch_u8

    def _proxy_nr_sharpness_quality(self, img_rgb, weights):
        """
        BRISQUE 不可用或 patch 退化时的代理值，只作兜底。
        此时 img_rgb / weights 已经是 ROI 域或全局域上的输入。
        """
        if img_rgb is None or weights is None:
            return 0.0
        if img_rgb.size == 0 or weights.size == 0:
            return 0.0

        patch_u8 = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
        patch_gray = cv2.cvtColor(patch_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        local_std = self._local_std_map(patch_gray, ksize=7)
        lap = np.abs(cv2.Laplacian(patch_gray, cv2.CV_32F))
        grad_x = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

        w = np.clip(weights.astype(np.float32), 0.0, 1.0)
        if float(w.sum()) <= 1e-8:
            return 0.0

        score = (
            0.4 * self._weighted_mean(local_std.reshape(-1), w.reshape(-1))
            + 0.3 * self._weighted_mean(lap.reshape(-1), w.reshape(-1))
            + 0.3 * self._weighted_mean(grad.reshape(-1), w.reshape(-1))
        )
        return self._safe_clip01(score / 0.12)

    def _golden_points(self, h, w):
        return np.array([
            [h * 0.382, w * 0.382],
            [h * 0.382, w * 0.618],
            [h * 0.618, w * 0.382],
            [h * 0.618, w * 0.618],
        ], dtype=np.float32)

    def _composition_score(self, weights):
        """
        原始设计更偏向“主体位置合理性”，不是整图恒定权重。
        """
        h, w = weights.shape
        s = float(np.sum(weights))
        if s <= 1e-8:
            return 0.0

        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cy = np.sum(ys * weights) / s
        cx = np.sum(xs * weights) / s

        pts = self._golden_points(h, w)
        dists = np.sqrt((pts[:, 0] - cy) ** 2 + (pts[:, 1] - cx) ** 2)
        min_dist = float(np.min(dists))
        diag = float(np.sqrt(h * h + w * w))
        score = 1.0 - min_dist / (0.35 * diag + 1e-8)
        return self._safe_clip01(score)

    def _build_global_subject_weights(self, semantic_mask, h, w):
        """
        global composition 使用硬 mask 的主体区域。
        优先 face / object / building；
        若不存在，则退回到除 background/noise 外的区域联合。
        """
        if semantic_mask is None:
            return np.ones((h, w), dtype=np.float32)

        weights = np.zeros((h, w), dtype=np.float32)

        for name in self.composition_subject_region_names:
            idx = self.region_name_to_idx.get(name, None)
            if idx is not None:
                weights += (semantic_mask == idx).astype(np.float32)

        if float(weights.sum()) <= 1e-8:
            for idx, name in SEMANTIC_LABELS.items():
                if name not in self.composition_fallback_exclude:
                    weights += (semantic_mask == idx).astype(np.float32)

        if float(weights.sum()) <= 1e-8:
            weights = np.ones((h, w), dtype=np.float32)

        return np.maximum(weights, 0.0)

    def _symmetry_score(self, img_gray, weights):
        patch = img_gray * np.clip(weights, 0.0, 1.0)
        h, w = patch.shape
        if min(h, w) < 16:
            return 0.0
        if float(patch.std()) < 0.02:
            return 0.0

        try:
            s_h = ssim(patch, patch[:, ::-1], data_range=1.0)
            s_v = ssim(patch, patch[::-1, :], data_range=1.0)
            return self._safe_clip01((float(s_h) + float(s_v)) * 0.5)
        except Exception:
            return 0.0

    def _nr_sharpness_quality(self, img_rgb, weights, use_true_brisque=False):
        """
        严格对齐版：
        - global 默认跑真实 BRISQUE
        - region 默认也跑真实 BRISQUE
        - 只有 patch 退化/异常时才回退 proxy
        """
        if not use_true_brisque:
            return self._proxy_nr_sharpness_quality(img_rgb, weights)

        patch_u8 = self._roi_masked_patch_for_brisque(img_rgb, weights)
        if patch_u8 is None:
            return self._proxy_nr_sharpness_quality(img_rgb, weights)

        patch_u8 = self._resize_if_needed(patch_u8, max_side=self.brisque_max_patch_side)

        if self._is_degenerate_patch(patch_u8):
            return self._proxy_nr_sharpness_quality(img_rgb, weights)

        if self.brisque_model is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    score = float(self.brisque_model.score(patch_u8))

                if not np.isfinite(score):
                    return self._proxy_nr_sharpness_quality(img_rgb, weights)

                return self._safe_clip01(1.0 - score / 100.0)
            except Exception:
                return self._proxy_nr_sharpness_quality(img_rgb, weights)

        if not self._warned_brisque:
            warnings.warn(
                "未使用真实 BRISQUE，nr_sharpness 将退化为局部清晰度代理值。",
                RuntimeWarning,
            )
            self._warned_brisque = True

        return self._proxy_nr_sharpness_quality(img_rgb, weights)

    def _color_cast_quality(self, img_rgb, img_lab, weights, prefix):
        """
        原始设计：
        - 人脸：YCbCr Cb/Cr 与标准值偏差率
        - 天空/植被等：Lab 与参考值欧氏距离
        - 最终都转成越高越好的质量值
        """
        w_flat = weights.reshape(-1).astype(np.float32)
        if float(w_flat.sum()) <= 1e-8:
            return 0.0

        if prefix == "face":
            img_ycrcb = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
            cr = img_ycrcb[..., 1].reshape(-1)
            cb = img_ycrcb[..., 2].reshape(-1)
            cr_mean = self._weighted_mean(cr, w_flat)
            cb_mean = self._weighted_mean(cb, w_flat)

            deviation = (abs(cr_mean - 128.0) + abs(cb_mean - 128.0)) / (2.0 * 127.0)
            return self._safe_clip01(1.0 - deviation)

        lab_a = img_lab[..., 1].reshape(-1) - 128.0
        lab_b = img_lab[..., 2].reshape(-1) - 128.0
        a_mean = self._weighted_mean(lab_a, w_flat)
        b_mean = self._weighted_mean(lab_b, w_flat)

        ref_map = {
            "sky": (0.0, -20.0),
            "vegetation": (-30.0, 60.0),
            "water": (0.0, 0.0),
            "building": (0.0, 0.0),
            "global": (0.0, 0.0),
            "background": (0.0, 0.0),
            "object": (0.0, 0.0),
            "noise": (0.0, 0.0),
        }
        ref_a, ref_b = ref_map.get(prefix, (0.0, 0.0))
        dist = np.sqrt((a_mean - ref_a) ** 2 + (b_mean - ref_b) ** 2) / 100.0
        return self._safe_clip01(1.0 - dist)

    def _texture_glcm_score(self, img_gray, weights):
        patch = np.clip(img_gray * np.clip(weights, 0.0, 1.0), 0.0, 1.0)
        h, w = patch.shape
        if min(h, w) < 12:
            return self._safe_clip01(float(patch.std()) / 0.15)

        levels = max(4, int(self.glcm_levels))
        patch_q = np.clip((patch * (levels - 1)).astype(np.uint8), 0, levels - 1)

        try:
            glcm = graycomatrix(
                patch_q,
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=levels,
                symmetric=True,
                normed=True,
            )
            contrast_mean = float(graycoprops(glcm, "contrast").mean())
            return self._safe_clip01(contrast_mean / 8.0)
        except Exception:
            return self._safe_clip01(float(patch.std()) / 0.15)

    def _michelson_contrast(self, img_gray, weights):
        vals = img_gray[weights > 1e-6]
        if vals.size == 0:
            return 0.0
        max_v = float(vals.max())
        min_v = float(vals.min())
        return self._safe_clip01((max_v - min_v) / (max_v + min_v + 1e-6))

    # ================= 6. 特征计算核心 =================
    def _extract_core_features_from_weights(
        self,
        img_rgb,
        img_gray,
        img_hsv,
        img_lab,
        lap_map,
        grad_map,
        weights,
        prefix,
        composition_weights=None,
    ):
        """
        返回的 12 维特征全部按“值越高越好”对齐原始设计。
        """
        weights = weights.astype(np.float32)
        weights = np.maximum(weights, 0.0)
        weight_sum = float(np.sum(weights))

        if weight_sum < float(self.min_region_pixels):
            return self._invalid_core_features(prefix), False, weight_sum

        gray_flat = img_gray.reshape(-1)
        sat_flat = img_hsv[..., 1].reshape(-1)
        lap_flat = lap_map.reshape(-1)
        grad_flat = grad_map.reshape(-1)
        w_flat = weights.reshape(-1)

        # 1) edge_sharpness：Laplacian + Sobel
        edge_raw = 0.5 * self._weighted_mean(lap_flat, w_flat) + 0.5 * self._weighted_mean(grad_flat, w_flat)
        edge_sharpness = self._safe_clip01(np.log1p(50.0 * edge_raw) / np.log(51.0))

        # 2) nr_sharpness：BRISQUE质量值
        use_true_brisque = self.use_true_brisque_for_global if prefix == "global" else self.use_true_brisque_for_regions
        nr_sharpness = self._nr_sharpness_quality(img_rgb, weights, use_true_brisque=use_true_brisque)

        # 3) color_cast：偏色质量值（越高越好）
        color_cast = self._color_cast_quality(img_rgb, img_lab, weights, prefix)

        # 4) saturation：HSV-S 均值
        saturation = self._safe_clip01(self._weighted_mean(sat_flat, w_flat))

        # 5) color_balance：HSV-S 方差取反
        sat_var = self._weighted_var(sat_flat, w_flat)
        color_balance = self._safe_clip01(1.0 - sat_var / 0.08)

        # 6) over_exposure：1 - 过爆率
        over_ratio = np.sum((gray_flat > 240.0 / 255.0).astype(np.float32) * w_flat) / (weight_sum + 1e-8)
        over_exposure = self._safe_clip01(1.0 - over_ratio)

        # 7) under_exposure：1 - 欠曝率
        under_ratio = np.sum((gray_flat < 15.0 / 255.0).astype(np.float32) * w_flat) / (weight_sum + 1e-8)
        under_exposure = self._safe_clip01(1.0 - under_ratio)

        # 8) brightness_uniformity：亮度方差取反
        gray_var = self._weighted_var(gray_flat, w_flat)
        brightness_uniformity = self._safe_clip01(1.0 - gray_var / 0.12)

        # 9) composition：主体位置合理性
        comp_w = composition_weights if composition_weights is not None else weights
        composition = self._composition_score(comp_w)

        # 10) symmetry：水平/垂直对称 SSIM
        symmetry = self._symmetry_score(img_gray, weights)

        # 11) texture：GLCM contrast 归一化
        texture = self._texture_glcm_score(img_gray, weights)

        # 12) contrast：Michelson 对比度
        contrast = self._michelson_contrast(img_gray, weights)

        feats = {
            f"{prefix}_edge_sharpness": float(edge_sharpness),
            f"{prefix}_nr_sharpness": float(nr_sharpness),
            f"{prefix}_color_cast": float(color_cast),
            f"{prefix}_saturation": float(saturation),
            f"{prefix}_color_balance": float(color_balance),
            f"{prefix}_over_exposure": float(over_exposure),
            f"{prefix}_under_exposure": float(under_exposure),
            f"{prefix}_brightness_uniformity": float(brightness_uniformity),
            f"{prefix}_composition": float(composition),
            f"{prefix}_symmetry": float(symmetry),
            f"{prefix}_texture": float(texture),
            f"{prefix}_contrast": float(contrast),
        }
        return feats, True, weight_sum

    # ================= 7. 单图处理 =================
    def extract_features_for_image(self, image_id, img_path, score_mapping):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None

        h, w = img_bgr.shape[:2]
        total_pixels = h * w

        img_rgb, img_gray, img_hsv, img_lab = self._prepare_images(img_bgr)
        lap_map, grad_map = self._compute_gradient_maps(img_gray)

        img_id_str = self._normalize_image_id(image_id)

        if img_id_str not in score_mapping:
            return None

        score_info = score_mapping[img_id_str]

        features = {
            "image_id": img_id_str,
            "label": float(score_info["label"]),
            "label_raw": float(score_info["label_raw"]),
        }

        semantic_mask, _ = self._load_hard_mask(img_id_str, h, w)
        global_comp_weights = self._build_global_subject_weights(semantic_mask, h, w)

        # A. 全局特征
        global_weights = np.ones((h, w), dtype=np.float32)
        global_feats, _, _ = self._extract_core_features_from_weights(
            img_rgb=img_rgb,
            img_gray=img_gray,
            img_hsv=img_hsv,
            img_lab=img_lab,
            lap_map=lap_map,
            grad_map=grad_map,
            weights=global_weights,
            prefix="global",
            composition_weights=global_comp_weights,
        )
        features.update(global_feats)

        # B. 区域特征（ROI-align优先；不再使用 soft-mask）
        if semantic_mask is None:
            for class_id in range(NUM_REGIONS):
                region_name = SEMANTIC_LABELS[class_id]
                features.update(self._invalid_core_features(region_name))
                features.update(self._region_meta(region_name, 0.0, total_pixels, is_valid=False))
        else:
            for class_id in range(NUM_REGIONS):
                region_name = SEMANTIC_LABELS[class_id]

                roi_payload, region_mass, is_valid = self._extract_region_roi_payload(
                    img_rgb=img_rgb,
                    img_gray=img_gray,
                    img_hsv=img_hsv,
                    img_lab=img_lab,
                    lap_map=lap_map,
                    grad_map=grad_map,
                    semantic_mask=semantic_mask,
                    class_id=class_id,
                    roi_size=self.roi_output_size,
                    margin_ratio=self.roi_margin_ratio,
                )

                if not is_valid or roi_payload is None:
                    features.update(self._invalid_core_features(region_name))
                    features.update(self._region_meta(region_name, region_mass, total_pixels, is_valid=False))
                    continue

                region_feats, _, _ = self._extract_core_features_from_weights(
                    img_rgb=roi_payload["img_rgb"],
                    img_gray=roi_payload["img_gray"],
                    img_hsv=roi_payload["img_hsv"],
                    img_lab=roi_payload["img_lab"],
                    lap_map=roi_payload["lap_map"],
                    grad_map=roi_payload["grad_map"],
                    weights=roi_payload["weights"],
                    prefix=region_name,
                    composition_weights=None,
                )

                features.update(region_feats)
                features.update(self._region_meta(region_name, region_mass, total_pixels, is_valid=True))

        return features

    # ================= 8. 列顺序（只保留单点评分） =================
    def _build_final_column_order(self):
        base_cols = ["image_id"]
        global_cols = [f"global_{name}" for name in CORE_FEATURE_ORDER]

        region_core_cols = []
        for region_name in REGION_ORDER:
            for feat_name in CORE_FEATURE_ORDER:
                region_core_cols.append(f"{region_name}_{feat_name}")

        region_meta_cols = []
        for region_name in REGION_ORDER:
            for meta_name in REGION_META_ORDER:
                region_meta_cols.append(f"{region_name}_{meta_name}")

        label_cols = ["label", "label_raw"]

        return base_cols + global_cols + region_core_cols + region_meta_cols + label_cols

    # ================= 9. 数据集处理 =================
    def process_dataset(self):
        print("🚀 开始特征提取（原始设计严格对齐版：12维全局 + 12维区域 + valid/area_ratio + ROI-align优先，无soft-mask，单点评分）...")

        if not _HAS_BRISQUE:
            print("⚠️ 当前未检测到 BRISQUE 包，nr_sharpness 将使用代理值。")
            print("   建议安装：pip install brisque libsvm-official")

        print(f"⚙️ BRISQUE策略: global={self.use_true_brisque_for_global}, region={self.use_true_brisque_for_regions}")
        print(f"⚙️ GLCM levels: {self.glcm_levels}")
        print(f"⚙️ ROI output size: {self.roi_output_size}")
        print(f"⚙️ ROI margin ratio: {self.roi_margin_ratio}")
        print(f"⚙️ Global composition subjects: {self.composition_subject_region_names}")

        score_mapping = self.load_ava_scores()
        if not score_mapping:
            print("❌ 未加载到评分数据，退出。")
            return

        final_cols = self._build_final_column_order()

        for split in ["train", "val", "test"]:
            split_file_candidates = [
                SPLIT_DIR / f"{split}.txt",
                SPLIT_DIR / f"{split}_ids.txt",
                RAW_DATA_DIR / f"{split}.txt",
            ]
            split_file = next((p for p in split_file_candidates if p.exists()), None)
            if split_file is None:
                print(f"⚠️ 缺少 split 文件: {split_file_candidates}")
                continue

            with open(split_file, "r") as f:
                image_ids = [self._normalize_image_id(line.strip()) for line in f if line.strip()]

            print(f"🔄 正在处理 {split} 集 ({len(image_ids)} 张)...")
            split_features = []
            missing_image_count = 0
            missing_label_count = 0

            for img_id in tqdm(image_ids):
                norm_id = self._normalize_image_id(img_id)

                if norm_id not in score_mapping:
                    missing_label_count += 1
                    continue

                img_path = self._resolve_image_path(img_id)
                if img_path is None:
                    missing_image_count += 1
                    continue

                feat = self.extract_features_for_image(img_id, img_path, score_mapping)
                if feat is not None:
                    split_features.append(feat)

            if not split_features:
                print(f"⚠️ {split} 未生成任何特征，跳过保存。")
                print(f"   - 缺失图像数量: {missing_image_count}")
                print(f"   - 缺失评分标签数量: {missing_label_count}")
                continue

            df = pd.DataFrame(split_features)

            missing_cols = [c for c in final_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少预期列，请检查代码逻辑: {missing_cols}")

            df = df[final_cols]

            save_path = FEATURE_DIR / f"{split}_features.csv"
            df.to_csv(save_path, index=False)

            global_core_dim = len(CORE_FEATURE_ORDER)
            region_core_dim = len(REGION_ORDER) * len(CORE_FEATURE_ORDER)
            region_meta_dim = len(REGION_ORDER) * len(REGION_META_ORDER)
            total_feature_dim = global_core_dim + region_core_dim + region_meta_dim

            print(f"✅ {split} 保存完毕: {save_path}")
            print(f"   - DataFrame 维度: {df.shape}")
            print(f"   - 全局特征维度: {global_core_dim}")
            print(f"   - 区域核心特征维度: {region_core_dim}")
            print(f"   - 区域元信息维度: {region_meta_dim}")
            print(f"   - 总手工特征维度: {total_feature_dim}")
            print(f"   - 缺失图像数量: {missing_image_count}")
            print(f"   - 缺失评分标签数量: {missing_label_count}")
            print(f"   - 固定区域顺序: {REGION_ORDER}")
            print(f"   - 核心特征顺序: {CORE_FEATURE_ORDER}")


if __name__ == "__main__":
    extractor = DesignAlignedFeatureExtractorROIHardMask(
        min_region_pixels=20,
        use_true_brisque_for_global=True,
        use_true_brisque_for_regions=True,
        brisque_min_patch_side=32,
        brisque_max_patch_side=256,
        glcm_levels=16,
        roi_output_size=(128, 128),
        roi_margin_ratio=0.05,
        composition_subject_region_names=("face", "object", "building"),
        composition_fallback_exclude=("background", "noise"),
    )
    extractor.process_dataset()
