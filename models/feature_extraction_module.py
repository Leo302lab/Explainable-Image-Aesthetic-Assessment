from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HandcraftedAestheticFeatureExtractor(nn.Module):
    """
    在线手工美学特征提取器。

    设计目标：
    1) 与训练阶段维度口径对齐：
       - global_handcrafted: [B, 12]
       - region_handcrafted: [B, R, 14] = 12 维 region_core + 2 维 meta(valid, area_ratio)
    2) 与 full_model.py 当前接口兼容：
       forward(image, region_soft_masks=None, region_padding_mask=None)
    3) 输入 image 默认视为 ImageNet 标准化后的张量，内部先反归一化到 [0,1] 后再提特征。

    注意：
    - 这版首先保证“维度与接口正确”，使训练好的 mapper / 评分头可以正常接入。
    - 若训练阶段离线 CSV 使用的 12 维具体定义与这里不同，数值分布可能仍有偏差；
      但这比旧版 7/9 维导致的线性层维度报错更接近你的训练口径。
    """

    def __init__(
        self,
        num_regions: int = 8,
        eps: float = 1e-6,
        imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.num_regions = int(num_regions)
        self.eps = float(eps)

        mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(imagenet_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

        lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("lap_kernel", lap_kernel, persistent=False)

    # ============================================================
    # 基础工具
    # ============================================================
    def _denorm_to_01(self, image: torch.Tensor) -> torch.Tensor:
        """将 ImageNet 标准化后的张量恢复到 [0,1]。"""
        image = image * self.imagenet_std.to(image.dtype) + self.imagenet_mean.to(image.dtype)
        return image.clamp(0.0, 1.0)

    def _grayscale(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B, 3, H, W] -> [B, 1, H, W]
        gray = (
            0.299 * image[:, 0:1, :, :]
            + 0.587 * image[:, 1:2, :, :]
            + 0.114 * image[:, 2:3, :, :]
        )
        return gray

    def _laplacian_response(self, gray: torch.Tensor) -> torch.Tensor:
        return F.conv2d(gray, self.lap_kernel.to(gray.dtype), padding=1)

    def _rgb_to_hsv_like(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        轻量近似 HSV，返回：hue[0,1], saturation[0,1], value[0,1]
        image: [B,3,H,W] in [0,1]
        """
        r = image[:, 0:1]
        g = image[:, 1:2]
        b = image[:, 2:3]

        maxc, _ = image.max(dim=1, keepdim=True)
        minc, _ = image.min(dim=1, keepdim=True)
        delta = maxc - minc

        # value
        v = maxc

        # saturation
        s = delta / (maxc + self.eps)

        # hue
        rc = (maxc - r) / (delta + self.eps)
        gc = (maxc - g) / (delta + self.eps)
        bc = (maxc - b) / (delta + self.eps)

        h = torch.zeros_like(maxc)
        r_mask = (maxc == r) & (delta > self.eps)
        g_mask = (maxc == g) & (delta > self.eps)
        b_mask = (maxc == b) & (delta > self.eps)

        h = torch.where(r_mask, (bc - gc) / 6.0, h)
        h = torch.where(g_mask, (2.0 + rc - bc) / 6.0, h)
        h = torch.where(b_mask, (4.0 + gc - rc) / 6.0, h)
        h = (h % 1.0).clamp(0.0, 1.0)

        return h, s.clamp(0.0, 1.0), v.clamp(0.0, 1.0)

    def _colorfulness_map(self, image: torch.Tensor) -> torch.Tensor:
        """
        Hasler & Süsstrunk 风格的简化 colorfulness map。
        返回 [B,1,H,W]。
        """
        r = image[:, 0:1]
        g = image[:, 1:2]
        b = image[:, 2:3]
        rg = r - g
        yb = 0.5 * (r + g) - b
        return torch.sqrt(rg.pow(2) + yb.pow(2) + self.eps)

    def _weighted_mean(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W], w: [B, R, H, W] -> [B, R]
        num = (x * w).sum(dim=(-1, -2))
        den = w.sum(dim=(-1, -2)).clamp_min(self.eps)
        return num / den

    def _weighted_var(self, x: torch.Tensor, w: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W], w:[B,R,H,W], mean:[B,R]
        diff2 = (x - mean.unsqueeze(-1).unsqueeze(-1)).pow(2)
        num = (diff2 * w).sum(dim=(-1, -2))
        den = w.sum(dim=(-1, -2)).clamp_min(self.eps)
        return num / den

    # ============================================================
    # 全局 / 区域核心特征（12 维）
    # ============================================================
    def _build_core_features_global(self, image_01: torch.Tensor) -> torch.Tensor:
        """
        输出 [B, 12]
        12 维定义：
        0 laplacian_abs_mean
        1 sharpness_rms
        2 saturation_mean
        3 saturation_std
        4 hue_mean
        5 hue_std
        6 brightness_mean
        7 brightness_std
        8 contrast_std(gray)
        9 over_expose_ratio
        10 under_expose_ratio
        11 colorfulness_mean
        """
        gray = self._grayscale(image_01)
        lap = self._laplacian_response(gray)
        hue, sat, value = self._rgb_to_hsv_like(image_01)
        colorfulness = self._colorfulness_map(image_01)

        laplacian_abs_mean = lap.abs().mean(dim=(-1, -2, -3))
        sharpness_rms = torch.sqrt(lap.pow(2).mean(dim=(-1, -2, -3)).clamp_min(self.eps))

        saturation_mean = sat.mean(dim=(-1, -2, -3))
        saturation_std = sat.flatten(1).std(dim=1, unbiased=False)

        hue_mean = hue.mean(dim=(-1, -2, -3))
        hue_std = hue.flatten(1).std(dim=1, unbiased=False)

        brightness_mean = value.mean(dim=(-1, -2, -3))
        brightness_std = value.flatten(1).std(dim=1, unbiased=False)

        contrast_std = gray.flatten(1).std(dim=1, unbiased=False)

        over_expose_ratio = (value > 0.95).float().mean(dim=(-1, -2, -3))
        under_expose_ratio = (value < 0.05).float().mean(dim=(-1, -2, -3))

        colorfulness_mean = colorfulness.mean(dim=(-1, -2, -3))

        return torch.stack(
            [
                laplacian_abs_mean,
                sharpness_rms,
                saturation_mean,
                saturation_std,
                hue_mean,
                hue_std,
                brightness_mean,
                brightness_std,
                contrast_std,
                over_expose_ratio,
                under_expose_ratio,
                colorfulness_mean,
            ],
            dim=-1,
        )

    def _build_core_features_region(
        self,
        image_01: torch.Tensor,
        region_soft_masks: Optional[torch.Tensor],
        region_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        输出 [B, R, 12] 的 region_core。
        与全局 12 维同定义，但按区域 soft mask 加权统计。
        """
        bsz = image_01.size(0)
        device = image_01.device
        dtype = image_01.dtype

        if region_soft_masks is None:
            return torch.zeros(bsz, self.num_regions, 12, device=device, dtype=dtype)

        gray = self._grayscale(image_01)
        lap = self._laplacian_response(gray)
        hue, sat, value = self._rgb_to_hsv_like(image_01)
        colorfulness = self._colorfulness_map(image_01)

        # [B,R]
        laplacian_abs_mean = self._weighted_mean(lap.abs(), region_soft_masks)
        sharpness_rms = torch.sqrt(
            self._weighted_mean(lap.pow(2), region_soft_masks).clamp_min(self.eps)
        )

        saturation_mean = self._weighted_mean(sat, region_soft_masks)
        saturation_var = self._weighted_var(sat, region_soft_masks, saturation_mean)
        saturation_std = torch.sqrt(saturation_var.clamp_min(self.eps))

        hue_mean = self._weighted_mean(hue, region_soft_masks)
        hue_var = self._weighted_var(hue, region_soft_masks, hue_mean)
        hue_std = torch.sqrt(hue_var.clamp_min(self.eps))

        brightness_mean = self._weighted_mean(value, region_soft_masks)
        brightness_var = self._weighted_var(value, region_soft_masks, brightness_mean)
        brightness_std = torch.sqrt(brightness_var.clamp_min(self.eps))

        gray_mean = self._weighted_mean(gray, region_soft_masks)
        gray_var = self._weighted_var(gray, region_soft_masks, gray_mean)
        contrast_std = torch.sqrt(gray_var.clamp_min(self.eps))

        over_expose_ratio = self._weighted_mean((value > 0.95).float(), region_soft_masks)
        under_expose_ratio = self._weighted_mean((value < 0.05).float(), region_soft_masks)

        colorfulness_mean = self._weighted_mean(colorfulness, region_soft_masks)

        region_core = torch.stack(
            [
                laplacian_abs_mean,
                sharpness_rms,
                saturation_mean,
                saturation_std,
                hue_mean,
                hue_std,
                brightness_mean,
                brightness_std,
                contrast_std,
                over_expose_ratio,
                under_expose_ratio,
                colorfulness_mean,
            ],
            dim=-1,
        )  # [B,R,12]

        if region_padding_mask is not None:
            region_core = region_core.masked_fill(region_padding_mask.unsqueeze(-1), 0.0)

        return region_core

    def _build_region_meta(
        self,
        region_soft_masks: Optional[torch.Tensor],
        region_padding_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        返回 [B, R, 2]: [valid, area_ratio]
        与训练 loader 中的 region_meta 口径对齐。 region_meta 是由 region_valid 和 region_area_ratio 拼出来的。fileciteturn32file4turn32file11
        """
        if region_soft_masks is None:
            return torch.zeros(batch_size, self.num_regions, 2, device=device, dtype=dtype)

        area_ratio = region_soft_masks.mean(dim=(-1, -2))
        valid = (area_ratio > 0).to(dtype)

        if region_padding_mask is not None:
            valid = valid.masked_fill(region_padding_mask, 0.0)
            area_ratio = area_ratio.masked_fill(region_padding_mask, 0.0)

        return torch.stack([valid, area_ratio], dim=-1)

    # ============================================================
    # 前向
    # ============================================================
    def forward(
        self,
        image: torch.Tensor,
        region_soft_masks: Optional[torch.Tensor] = None,
        region_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # 先反归一化到 [0,1]
        image_01 = self._denorm_to_01(image)

        global_core = self._build_core_features_global(image_01)  # [B,12]
        region_core = self._build_core_features_region(
            image_01=image_01,
            region_soft_masks=region_soft_masks,
            region_padding_mask=region_padding_mask,
        )  # [B,R,12]

        region_meta = self._build_region_meta(
            region_soft_masks=region_soft_masks,
            region_padding_mask=region_padding_mask,
            batch_size=image.size(0),
            device=image.device,
            dtype=image.dtype,
        )  # [B,R,2]

        region_handcrafted = torch.cat([region_core, region_meta], dim=-1)  # [B,R,14]

        # 运行期断言，尽早暴露维度不一致问题
        if global_core.size(-1) != 12:
            raise ValueError(f"global_handcrafted dim mismatch: got {global_core.size(-1)}, expected 12")
        if region_handcrafted.size(-1) != 14:
            raise ValueError(f"region_handcrafted dim mismatch: got {region_handcrafted.size(-1)}, expected 14")

        return {
            "global_handcrafted": global_core,
            "region_handcrafted": region_handcrafted,
            # 额外返回，便于调试 / 与训练期 loader 对齐检查
            "region_core": region_core,
            "region_meta": region_meta,
        }


# ============================================================
# 旧接口兼容别名
# ============================================================
HandcraftedFeatureExtractor = HandcraftedAestheticFeatureExtractor
AestheticFeatureExtractor = HandcraftedAestheticFeatureExtractor
FeatureExtractionModule = HandcraftedAestheticFeatureExtractor
