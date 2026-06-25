from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.ops import roi_align
from transformers import SegformerForSemanticSegmentation, ViTConfig, ViTModel

from utils.config import ModelConfig, denormalize_score
from models.cross_attention_fusion import CrossAttentionFusion
from models.feature_extraction_module import HandcraftedAestheticFeatureExtractor
from models.score_prediction_module import ScorePredictionModule


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    return ckpt


def _strip_prefix_levels(state_dict, levels):
    new_state_dict = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        new_key = ".".join(parts[levels:]) if len(parts) > levels else key
        new_state_dict[new_key] = value
    return new_state_dict


def _align_state_dict_to_module(module: nn.Module, raw_state_dict):
    target_keys = set(module.state_dict().keys())
    cleaned = {key.replace("module.", ""): value for key, value in raw_state_dict.items()}

    best_overlap = -1
    best_state_dict = cleaned
    for levels in range(5):
        candidate = cleaned if levels == 0 else _strip_prefix_levels(cleaned, levels)
        overlap = len(set(candidate.keys()) & target_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_state_dict = candidate

    filtered = {k: v for k, v in best_state_dict.items() if k in target_keys}
    if filtered:
        return filtered, best_overlap
    return best_state_dict, best_overlap


def _safe_load_module(module: nn.Module, ckpt_path: Optional[str], name: str = "module") -> None:
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        print(f"⚠️ [{name}] checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    state_dict, matched = _align_state_dict_to_module(module, state_dict)

    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    print(f"✅ [{name}] loaded checkpoint: {ckpt_path}")
    print(f"   matched keys: {matched}")
    if missing:
        print(f"   missing keys: {len(missing)}")
    if unexpected:
        print(f"   unexpected keys: {len(unexpected)}")


class FullAestheticModel(nn.Module):
    """
    最小修改版完整模型：
    - 不改训练主流程
    - 新增 handcrafted -> final_score 的可复算接口，供解释分支做 final_score SHAP / 显式消融
    - 当前 region_contributions 继续保留，但仅作为调试量，不再建议作为最终区域贡献
    """

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()

        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg
        self.num_regions = int(cfg.num_regions)
        self.score_range = cfg.score_range

        self.use_roi_align_region_features = bool(getattr(cfg, "use_roi_align_region_features", True))
        self.roi_output_size = int(getattr(cfg, "roi_output_size", 7))
        self.roi_sampling_ratio = int(getattr(cfg, "roi_sampling_ratio", 2))
        self.roi_align_binarize_threshold = float(getattr(cfg, "roi_align_binarize_threshold", 0.5))
        self.roi_min_box_size = float(getattr(cfg, "roi_min_box_size", 1.0))

        self.seg_branch = SegformerForSemanticSegmentation.from_pretrained(
            cfg.seg_model_name_or_path,
            num_labels=cfg.num_regions,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        _safe_load_module(self.seg_branch, getattr(cfg, "seg_finetuned_ckpt", None), name="SegFormer-Finetuned")
        if getattr(cfg, "freeze_seg", False):
            for p in self.seg_branch.parameters():
                p.requires_grad = False

        backbone = resnet50(weights=None)
        _safe_load_module(backbone, getattr(cfg, "resnet50_ckpt", None), name="ResNet50")
        self.layer3 = nn.Sequential(*list(backbone.children())[:-3])
        self.layer4 = nn.Sequential(*list(backbone.children())[-3:-2])
        self.cnn_backbone = nn.Sequential(self.layer3, self.layer4)
        self.cnn_out_dim = 2048
        self.layer3_out_dim = 1024

        self.layer3_projector = nn.Sequential(
            nn.Conv2d(self.layer3_out_dim, cfg.region_feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.region_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_projector = nn.Sequential(
            nn.Conv2d(self.cnn_out_dim, cfg.region_feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.region_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.multiscale_fusion = nn.Sequential(
            nn.Conv2d(cfg.region_feature_dim * 2, cfg.region_feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.region_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.region_roi_encoder = nn.Sequential(
            nn.Conv2d(cfg.region_feature_dim, cfg.region_feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.region_feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        try:
            self.vit = ViTModel.from_pretrained(cfg.vit_model_name_or_path, local_files_only=True)
        except Exception:
            vit_cfg = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
            )
            self.vit = ViTModel(vit_cfg)

        _safe_load_module(self.vit, getattr(cfg, "vit_ckpt_path", None), name="ViT-Base")
        if getattr(cfg, "freeze_vit", False):
            for p in self.vit.parameters():
                p.requires_grad = False

        self.patch_projector = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, cfg.region_feature_dim),
            nn.LayerNorm(cfg.region_feature_dim),
        )
        self.global_projector = nn.Sequential(
            nn.Linear(self.cnn_out_dim + self.vit.config.hidden_size, cfg.global_feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(cfg.global_feature_dim),
        )

        self.global_hand_mapper = nn.Sequential(
            nn.Linear(cfg.handcrafted_global_dim, cfg.global_hand_embed_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.global_hand_embed_dim),
        )
        self.region_hand_mapper = nn.Sequential(
            nn.Linear(cfg.handcrafted_region_dim, cfg.region_hand_embed_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.region_hand_embed_dim),
        )
        self.region_hand_align_mapper = nn.Sequential(
            nn.Linear(cfg.handcrafted_region_dim, cfg.region_feature_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.region_feature_dim),
        )
        self.global_fuser = nn.Sequential(
            nn.Linear(cfg.global_feature_dim + cfg.global_hand_embed_dim, cfg.global_feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(cfg.global_feature_dim),
        )
        self.region_fuser = nn.Sequential(
            nn.Linear(cfg.region_feature_dim + cfg.region_hand_embed_dim, cfg.region_feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(cfg.region_feature_dim),
        )
        self.handcrafted_extractor = HandcraftedAestheticFeatureExtractor(num_regions=cfg.num_regions)

        self.fusion_global_adapter = nn.Sequential(
            nn.Linear(cfg.global_feature_dim, cfg.region_feature_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.region_feature_dim),
        )
        self.region_importance = nn.Sequential(
            nn.Linear(cfg.region_feature_dim, 1),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Linear(cfg.global_feature_dim + cfg.global_feature_dim, cfg.global_feature_dim),
            nn.Sigmoid(),
        )
        self.cross_attention_fusion = CrossAttentionFusion(
            query_dim=cfg.region_feature_dim,
            kv_dim=cfg.region_feature_dim,
            hidden_dim=cfg.region_feature_dim,
            num_heads=cfg.cross_attn_heads,
            dropout=cfg.cross_attn_dropout,
            output_dim=cfg.global_feature_dim,
        )
        self.prediction_module = ScorePredictionModule(
            global_feature_dim=cfg.global_feature_dim,
            region_feature_dim=cfg.region_feature_dim,
            handcrafted_global_dim=cfg.handcrafted_global_dim,
            handcrafted_region_dim=cfg.handcrafted_region_dim,
            global_hidden_dim=cfg.global_score_hidden_dim,
            region_hidden_dim=cfg.region_score_hidden_dim,
            num_regions=cfg.num_regions,
            region_type_embed_dim=getattr(cfg, "region_type_embed_dim", 0),
        )

    def extract_vit_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vit_out = self.vit(pixel_values=x)
        tokens = vit_out.last_hidden_state
        vit_cls = tokens[:, 0]
        patch_tokens = self.patch_projector(tokens[:, 1:])
        return vit_cls, patch_tokens

    @staticmethod
    def _normalize_region_soft_masks(region_soft_masks: torch.Tensor) -> torch.Tensor:
        mask_sum = region_soft_masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return region_soft_masks / mask_sum

    def _convert_region_masks_to_channel_first(self, region_masks: torch.Tensor) -> torch.Tensor:
        if region_masks.dim() == 3:
            region_masks = F.one_hot(region_masks.long(), num_classes=self.cfg.num_regions).permute(0, 3, 1, 2).float()
        elif region_masks.dim() == 4 and region_masks.shape[1] == 1:
            region_masks = F.one_hot(region_masks[:, 0].long(), num_classes=self.cfg.num_regions).permute(0, 3, 1, 2).float()
        elif region_masks.dim() == 4 and region_masks.shape[1] == self.cfg.num_regions:
            region_masks = region_masks.float()
        elif region_masks.dim() == 4 and region_masks.shape[-1] == self.cfg.num_regions:
            region_masks = region_masks.permute(0, 3, 1, 2).float()
        else:
            raise ValueError(f"Unsupported seg_masks shape: {tuple(region_masks.shape)}")
        return region_masks

    def _build_image_region_soft_masks_from_logits(self, seg_logits: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        logits_up = F.interpolate(seg_logits, size=image_hw, mode="bilinear", align_corners=False)
        return torch.softmax(logits_up, dim=1)

    def _build_image_region_soft_masks_from_masks(self, region_masks: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        region_masks_cf = self._convert_region_masks_to_channel_first(region_masks)
        region_soft_masks = F.interpolate(region_masks_cf, size=image_hw, mode="bilinear", align_corners=False)
        return self._normalize_region_soft_masks(region_soft_masks)

    def _build_feature_level_region_soft_masks_from_masks(self, region_masks: torch.Tensor, feature_hw: Tuple[int, int]) -> torch.Tensor:
        region_masks_cf = self._convert_region_masks_to_channel_first(region_masks)
        region_soft_masks = F.interpolate(region_masks_cf, size=feature_hw, mode="bilinear", align_corners=False)
        return self._normalize_region_soft_masks(region_soft_masks)

    def _extract_boxes_from_feature_masks(self, region_hard_masks: torch.Tensor, min_region_mass: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if region_hard_masks.dim() != 4:
            raise ValueError(f"Expected region_hard_masks [B,R,H,W], got {tuple(region_hard_masks.shape)}")

        B, R, Hf, Wf = region_hard_masks.shape
        device = region_hard_masks.device
        region_mass = region_hard_masks.flatten(2).sum(dim=-1)
        region_padding_mask = region_mass < float(min_region_mass)

        roi_boxes = []
        for b in range(B):
            for r in range(R):
                if region_padding_mask[b, r]:
                    roi_boxes.append(torch.tensor([b, 0.0, 0.0, self.roi_min_box_size, self.roi_min_box_size], device=device))
                    continue

                mask = region_hard_masks[b, r] > self.roi_align_binarize_threshold
                coords = torch.where(mask)
                if coords[0].numel() == 0:
                    region_padding_mask[b, r] = True
                    roi_boxes.append(torch.tensor([b, 0.0, 0.0, self.roi_min_box_size, self.roi_min_box_size], device=device))
                    continue

                y_min = coords[0].min().float()
                x_min = coords[1].min().float()
                y_max = coords[0].max().float()
                x_max = coords[1].max().float()

                if (x_max - x_min) < self.roi_min_box_size:
                    x_max = x_min + self.roi_min_box_size
                if (y_max - y_min) < self.roi_min_box_size:
                    y_max = y_min + self.roi_min_box_size

                x_max = torch.clamp(x_max, max=float(Wf - 1))
                y_max = torch.clamp(y_max, max=float(Hf - 1))
                roi_boxes.append(torch.tensor([b, x_min, y_min, x_max, y_max], device=device))

        return torch.stack(roi_boxes, dim=0).float(), region_padding_mask

    def _roi_align_region_features(self, spatial_feats: torch.Tensor, roi_boxes_flat: torch.Tensor, region_padding_mask: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = spatial_feats.shape
        R = region_padding_mask.shape[1]
        roi_feats = roi_align(
            input=spatial_feats,
            boxes=roi_boxes_flat,
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=1.0,
            sampling_ratio=self.roi_sampling_ratio,
            aligned=True,
        )
        region_feats = self.region_roi_encoder(roi_feats).view(B, R, C)
        region_feats = region_feats.masked_fill(region_padding_mask.unsqueeze(-1), 0.0)
        return region_feats

    def _extract_region_features_mask_pooling(self, spatial_feats: torch.Tensor, region_soft_masks: torch.Tensor, min_region_mass: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        region_mass = region_soft_masks.flatten(2).sum(dim=-1)
        region_padding_mask = region_mass < min_region_mass
        norm_masks = region_soft_masks / (region_mass.unsqueeze(-1).unsqueeze(-1) + 1e-6)
        region_feats = torch.einsum("brhw,bchw->brc", norm_masks, spatial_feats)
        region_feats = region_feats.masked_fill(region_padding_mask.unsqueeze(-1), 0.0)
        return region_feats, region_padding_mask, region_soft_masks

    def extract_region_features(self, spatial_feats: torch.Tensor, seg_logits: torch.Tensor, min_region_mass: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if min_region_mass is None:
            min_region_mass = self.cfg.min_region_mass

        _, _, height, width = spatial_feats.shape
        logits_up = F.interpolate(seg_logits, size=(height, width), mode="bilinear", align_corners=False)
        region_soft_masks = torch.softmax(logits_up, dim=1)

        if not self.use_roi_align_region_features:
            return self._extract_region_features_mask_pooling(spatial_feats, region_soft_masks, min_region_mass)

        hard_label = torch.argmax(logits_up, dim=1)
        region_hard_masks = F.one_hot(hard_label.long(), num_classes=self.cfg.num_regions).permute(0, 3, 1, 2).float()
        roi_boxes_flat, region_padding_mask = self._extract_boxes_from_feature_masks(region_hard_masks, min_region_mass)
        region_feats = self._roi_align_region_features(spatial_feats, roi_boxes_flat, region_padding_mask)
        return region_feats, region_padding_mask, region_soft_masks

    def extract_region_features_from_masks(self, spatial_feats: torch.Tensor, region_masks: torch.Tensor, min_region_mass: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if min_region_mass is None:
            min_region_mass = self.cfg.min_region_mass

        _, _, height, width = spatial_feats.shape
        region_soft_masks = self._build_feature_level_region_soft_masks_from_masks(region_masks, feature_hw=(height, width))
        if not self.use_roi_align_region_features:
            return self._extract_region_features_mask_pooling(spatial_feats, region_soft_masks, min_region_mass)

        hard_label = torch.argmax(region_soft_masks, dim=1)
        region_hard_masks = F.one_hot(hard_label.long(), num_classes=self.cfg.num_regions).permute(0, 3, 1, 2).float()
        roi_boxes_flat, region_padding_mask = self._extract_boxes_from_feature_masks(region_hard_masks, min_region_mass)
        region_feats = self._roi_align_region_features(spatial_feats, roi_boxes_flat, region_padding_mask)
        return region_feats, region_padding_mask, region_soft_masks

    def _apply_region_dropout(self, region_features: torch.Tensor, region_padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.cfg.region_dropout_prob <= 0:
            return region_features, region_padding_mask

        keep_mask = torch.rand(region_padding_mask.shape, device=region_padding_mask.device) > self.cfg.region_dropout_prob
        keep_mask = keep_mask | (~region_padding_mask)
        new_padding_mask = ~keep_mask
        region_features = region_features.masked_fill(new_padding_mask.unsqueeze(-1), 0.0)
        return region_features, new_padding_mask

    def _fuse_handcrafted_features(
        self,
        deep_global: torch.Tensor,
        deep_regions: Optional[torch.Tensor],
        handcrafted_global: Optional[torch.Tensor],
        handcrafted_region: Optional[torch.Tensor],
        region_padding_mask: Optional[torch.Tensor],
        apply_region_dropout: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        mapped_global_hand = None
        mapped_region_hand = None
        align_region_hand = None
        pre_region_importance = None

        fused_global = deep_global
        fused_regions = deep_regions

        if handcrafted_global is not None:
            mapped_global_hand = self.global_hand_mapper(handcrafted_global)
            fused_global = self.global_fuser(torch.cat([deep_global, mapped_global_hand], dim=-1))

        if deep_regions is not None and handcrafted_region is not None:
            mapped_region_hand = self.region_hand_mapper(handcrafted_region)
            align_region_hand = self.region_hand_align_mapper(handcrafted_region)
            fused_regions = self.region_fuser(torch.cat([deep_regions, mapped_region_hand], dim=-1))
            pre_region_importance = self.region_importance(fused_regions)
            fused_regions = fused_regions * pre_region_importance.expand_as(fused_regions)

        if apply_region_dropout and fused_regions is not None and region_padding_mask is not None:
            fused_regions, region_padding_mask = self._apply_region_dropout(fused_regions, region_padding_mask)

        return {
            "fused_global": fused_global,
            "fused_regions": fused_regions,
            "mapped_global_handcrafted": mapped_global_hand,
            "mapped_region_handcrafted": mapped_region_hand,
            "region_hand_align_feature": align_region_hand,
            "pre_region_importance": pre_region_importance,
            "region_padding_mask": region_padding_mask,
        }

    def _run_fusion_and_prediction(
        self,
        fused_global: torch.Tensor,
        fused_regions: Optional[torch.Tensor],
        patch_tokens: torch.Tensor,
        handcrafted_global: Optional[torch.Tensor],
        handcrafted_region: Optional[torch.Tensor],
        region_padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        cross_attn_weights = None
        aggregated_region_attention_weights = None
        head_importance = None
        context_regions_hidden = None
        fused_feature_before_gate = None

        if fused_regions is not None:
            if not getattr(self.cfg, "ablate_cross_attention", False):
                if patch_tokens is None:
                    raise ValueError("patch_tokens is required when regions are enabled and cross-attention is active.")

                fusion_global = self.fusion_global_adapter(fused_global)
                fusion_out = self.cross_attention_fusion(
                    global_feature=fusion_global,
                    global_patch_tokens=patch_tokens,
                    region_features=fused_regions,
                    region_padding_mask=region_padding_mask,
                )

                context_regions = fusion_out["context_region_features_hidden"]
                context_regions_hidden = fusion_out.get("context_region_features_hidden")
                region_weights = fusion_out["region_importance"]

                fused_feature_before_gate = fusion_out["fused_feature"]
                gate_value = self.gate(torch.cat([fused_global, fused_feature_before_gate], dim=-1))
                fused_feature = gate_value * fused_feature_before_gate + (1.0 - gate_value) * fused_global

                cross_attn_weights = fusion_out.get("cross_attn_weights")
                head_importance = fusion_out.get("head_importance")
                if cross_attn_weights is not None:
                    if cross_attn_weights.dim() == 4:
                        region_level = cross_attn_weights.mean(dim=-1)  # [B,H,R]
                        if head_importance is not None and head_importance.dim() == 2:
                            aggregated_region_attention_weights = (
                                region_level * head_importance.unsqueeze(-1)
                            ).sum(dim=1)
                        else:
                            aggregated_region_attention_weights = region_level.mean(dim=1)
                    elif cross_attn_weights.dim() == 3:
                        if head_importance is not None and head_importance.dim() == 2:
                            aggregated_region_attention_weights = (
                                cross_attn_weights * head_importance.unsqueeze(-1)
                            ).sum(dim=1)
                        else:
                            aggregated_region_attention_weights = cross_attn_weights.mean(dim=1)
                    elif cross_attn_weights.dim() == 2:
                        aggregated_region_attention_weights = cross_attn_weights
            else:
                context_regions = fused_regions
                region_weights = None
                fused_feature = fused_global
        else:
            context_regions = None
            region_weights = None
            fused_feature = fused_global

        pred_out = self.prediction_module(
            global_deep_feature=fused_feature,
            region_deep_features=context_regions,
            region_weights=region_weights,
            handcrafted_global_feature=handcrafted_global,
            handcrafted_region_features=handcrafted_region,
            region_padding_mask=region_padding_mask,
        )

        region_contributions_debug = pred_out.get("region_logit_contributions")
        region_scores_for_contrib = pred_out.get("region_scores_raw", pred_out.get("region_scores"))
        if (
            region_contributions_debug is None
            and region_scores_for_contrib is not None
            and pred_out["region_weights_normalized"] is not None
        ):
            region_contributions_debug = pred_out["region_weights_normalized"] * (region_scores_for_contrib - 0.5)
            if region_padding_mask is not None:
                region_contributions_debug = region_contributions_debug.masked_fill(region_padding_mask, 0.0)

        return {
            "prediction": pred_out,
            "context_region_features": context_regions,
            "context_region_features_hidden": context_regions_hidden,
            "region_weights": region_weights,
            "cross_attn_weights": cross_attn_weights,
            "aggregated_region_attention_weights": aggregated_region_attention_weights,
            "head_importance": head_importance,
            "fused_feature": fused_feature,
            "fused_feature_before_gate": fused_feature_before_gate,
            "region_contributions_debug": region_contributions_debug,
        }

    def forward_from_intermediate_features(
        self,
        global_deep_feature: torch.Tensor,
        region_deep_features: Optional[torch.Tensor],
        patch_tokens: torch.Tensor,
        handcrafted_global: Optional[torch.Tensor] = None,
        handcrafted_region: Optional[torch.Tensor] = None,
        region_padding_mask: Optional[torch.Tensor] = None,
        image_region_soft_masks: Optional[torch.Tensor] = None,
        region_soft_masks: Optional[torch.Tensor] = None,
        apply_region_dropout: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        fused_pack = self._fuse_handcrafted_features(
            deep_global=global_deep_feature,
            deep_regions=region_deep_features,
            handcrafted_global=handcrafted_global,
            handcrafted_region=handcrafted_region,
            region_padding_mask=region_padding_mask,
            apply_region_dropout=apply_region_dropout,
        )
        run_pack = self._run_fusion_and_prediction(
            fused_global=fused_pack["fused_global"],
            fused_regions=fused_pack["fused_regions"],
            patch_tokens=patch_tokens,
            handcrafted_global=handcrafted_global,
            handcrafted_region=handcrafted_region,
            region_padding_mask=fused_pack["region_padding_mask"],
        )
        pred_out = run_pack["prediction"]
        return {
            "final_score": pred_out["final_score"],
            "final_score_10": denormalize_score(pred_out["final_score"], self.score_range),
            "final_logit": pred_out["final_logit"],
            "global_score": pred_out["global_score"],
            "global_score_10": denormalize_score(pred_out["global_score"], self.score_range),
            "global_logit": pred_out["global_logit"],
            "region_scores": pred_out["region_scores"],
            "region_scores_10": None if pred_out["region_scores"] is None else denormalize_score(pred_out["region_scores"], self.score_range),
            "region_logits": pred_out["region_logits"],
            "region_logits_raw": pred_out.get("region_logits_raw"),
            "region_scores_raw": pred_out.get("region_scores_raw"),
            "region_valid_mask": pred_out.get("region_valid_mask"),
            "region_weights": pred_out.get("region_weights_normalized"),
            "prediction_region_weights_normalized": pred_out.get("prediction_region_weights_normalized"),
            "region_attention_weights": run_pack["cross_attn_weights"],
            "aggregated_region_attention_weights": run_pack["aggregated_region_attention_weights"],
            "head_importance": run_pack["head_importance"],
            "region_agg_logit": pred_out["region_agg_logit"],
            "region_logit_contributions": pred_out.get("region_logit_contributions"),
            "region_contributions": run_pack["region_contributions_debug"],
            "region_contributions_debug": run_pack["region_contributions_debug"],
            "fusion_gate": pred_out.get("fusion_gate"),
            "region_padding_mask": fused_pack["region_padding_mask"],
            "image_region_soft_masks": image_region_soft_masks,
            "region_soft_masks": region_soft_masks,
            "deep_global_feature": global_deep_feature,
            "deep_region_features": region_deep_features,
            "fused_global_feature": fused_pack["fused_global"],
            "fused_region_features": fused_pack["fused_regions"],
            "fused_feature": run_pack["fused_feature"],
            "fused_feature_before_gate": run_pack["fused_feature_before_gate"],
            "context_region_features": run_pack["context_region_features"],
            "context_region_features_hidden": run_pack["context_region_features_hidden"],
            "cross_attn_weights": run_pack["cross_attn_weights"],
            "mapped_global_handcrafted": fused_pack["mapped_global_handcrafted"],
            "mapped_region_handcrafted": fused_pack["mapped_region_handcrafted"],
            "region_hand_align_feature": fused_pack["region_hand_align_feature"],
            "pre_region_importance": fused_pack["pre_region_importance"],
            "handcrafted_global_features": handcrafted_global,
            "handcrafted_region_features": handcrafted_region,
            "handcrafted_global_raw": handcrafted_global,
            "handcrafted_region_raw": handcrafted_region,
            "raw_handcrafted_global": handcrafted_global,
            "raw_handcrafted_region": handcrafted_region,
            "patch_tokens": patch_tokens,
        }

    def forward(
        self,
        x: torch.Tensor,
        handcrafted_global: Optional[torch.Tensor] = None,
        handcrafted_region: Optional[torch.Tensor] = None,
        x_vit: Optional[torch.Tensor] = None,
        seg_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        layer3_feats = self.layer3(x)
        layer4_feats = self.layer4(layer3_feats)
        cnn_spatial = layer4_feats
        cnn_global = F.adaptive_avg_pool2d(cnn_spatial, output_size=(1, 1)).flatten(1)

        layer3_proj = self.layer3_projector(layer3_feats)
        layer4_proj = self.spatial_projector(layer4_feats)
        layer3_proj = F.interpolate(layer3_proj, size=layer4_proj.shape[2:], mode="bilinear", align_corners=False)
        spatial_proj = self.multiscale_fusion(torch.cat([layer3_proj, layer4_proj], dim=1))

        if x_vit is None:
            x_vit = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)

        vit_cls, patch_tokens = self.extract_vit_tokens(x_vit)
        deep_global = self.global_projector(torch.cat([cnn_global, vit_cls], dim=-1))

        image_h, image_w = x.shape[-2], x.shape[-1]
        if getattr(self.cfg, "disable_regions", False) or getattr(self.cfg, "ablate_segmentation", False):
            seg_logits = None
            region_soft_masks = None
            image_region_soft_masks = None
            deep_regions = None
            region_padding_mask = None
        else:
            if seg_masks is not None:
                seg_logits = None
                deep_regions, region_padding_mask, region_soft_masks = self.extract_region_features_from_masks(
                    spatial_feats=spatial_proj,
                    region_masks=seg_masks,
                )
                image_region_soft_masks = self._build_image_region_soft_masks_from_masks(seg_masks, image_hw=(image_h, image_w))
            else:
                seg_logits = self.seg_branch(pixel_values=x).logits
                deep_regions, region_padding_mask, region_soft_masks = self.extract_region_features(
                    spatial_feats=spatial_proj,
                    seg_logits=seg_logits,
                )
                image_region_soft_masks = self._build_image_region_soft_masks_from_logits(seg_logits, image_hw=(image_h, image_w))

        if self.cfg.use_online_handcrafted_extractor and (
            handcrafted_global is None or (deep_regions is not None and handcrafted_region is None)
        ):
            extracted = self.handcrafted_extractor(
                image=x,
                region_soft_masks=image_region_soft_masks if deep_regions is not None else None,
                region_padding_mask=region_padding_mask if deep_regions is not None else None,
            )
            if handcrafted_global is None:
                handcrafted_global = extracted["global_handcrafted"]
            if deep_regions is not None and handcrafted_region is None:
                handcrafted_region = extracted["region_handcrafted"]

        forward_pack = self.forward_from_intermediate_features(
            global_deep_feature=deep_global,
            region_deep_features=deep_regions,
            patch_tokens=patch_tokens,
            handcrafted_global=handcrafted_global,
            handcrafted_region=handcrafted_region,
            region_padding_mask=region_padding_mask,
            image_region_soft_masks=image_region_soft_masks,
            region_soft_masks=region_soft_masks,
            apply_region_dropout=True,
        )

        return {
            **forward_pack,
            "seg_logits": seg_logits,
        }


SSGCAFModel = FullAestheticModel
