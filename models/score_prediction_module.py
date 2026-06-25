from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScorePredictionModule(nn.Module):
    """
    对齐原始设计的评分头：

    1. 区域子评分 = 区域特征 + 区域注意力权重 -> region_score_head
    2. padding 区域不输出伪中间分，而是显式 NaN
    3. handcrafted 缺失时自动补零，保持输入维度稳定
    4. 保留 region_type_embedding 与 hand-only 诊断头
    5. region_logit_contributions 仅作为调试量，不作为最终解释依据
    """

    def __init__(
        self,
        global_feature_dim: int,
        region_feature_dim: int,
        handcrafted_global_dim: int = 0,
        handcrafted_region_dim: int = 0,
        global_hidden_dim: int = 64,
        region_hidden_dim: int = 64,
        hand_global_embed_dim: int = 64,
        hand_region_embed_dim: int = 32,
        dropout: float = 0.5,
        num_regions: Optional[int] = None,
        region_type_embed_dim: int = 0,
    ) -> None:
        super().__init__()
        self.region_enabled = True

        self.global_feature_dim = int(global_feature_dim)
        self.region_feature_dim = int(region_feature_dim)
        self.handcrafted_global_dim = int(handcrafted_global_dim)
        self.handcrafted_region_dim = int(handcrafted_region_dim)
        self.num_regions = num_regions
        self.region_type_embed_dim = int(region_type_embed_dim)
        self.hand_global_embed_dim = int(hand_global_embed_dim if handcrafted_global_dim > 0 else 0)
        self.hand_region_embed_dim = int(hand_region_embed_dim if handcrafted_region_dim > 0 else 0)

        self.global_hand_embed = None
        if handcrafted_global_dim > 0:
            self.global_hand_embed = MLP(
                handcrafted_global_dim,
                hand_global_embed_dim,
                hand_global_embed_dim,
                dropout=dropout,
            )

        self.region_hand_embed = None
        if handcrafted_region_dim > 0:
            self.region_hand_embed = MLP(
                handcrafted_region_dim,
                hand_region_embed_dim,
                hand_region_embed_dim,
                dropout=dropout,
            )

        self.region_type_embed = None
        if region_type_embed_dim > 0:
            if num_regions is None or num_regions <= 0:
                raise ValueError("When region_type_embed_dim > 0, num_regions must be a positive integer.")
            self.region_type_embed = nn.Embedding(num_regions, region_type_embed_dim)

        global_in_dim = global_feature_dim + self.hand_global_embed_dim

        # 关键改动：区域子评分头显式吃到 region weight（+1）
        region_in_dim = region_feature_dim + self.hand_region_embed_dim + self.region_type_embed_dim + 1

        self.global_score_head = MLP(global_in_dim, global_hidden_dim, 1, dropout=dropout)
        self.region_score_head = MLP(region_in_dim, region_hidden_dim, 1, dropout=dropout)

        self.region_hand_only_head = None
        if handcrafted_region_dim > 0:
            hand_only_in_dim = self.hand_region_embed_dim + self.region_type_embed_dim
            self.region_hand_only_head = MLP(hand_only_in_dim, region_hidden_dim, 1, dropout=dropout)

        self.region_agg_head = nn.Sequential(
            nn.Linear(1, region_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(region_hidden_dim, 1),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(2, global_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden_dim, 1),
        )

        self.final_fusion = nn.Sequential(
            nn.Linear(2, global_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden_dim, 1),
        )

    @staticmethod
    def _prepare_region_weights(weights: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if weights is None:
            return None
        if weights.dim() == 3 and weights.shape[-1] == 1:
            weights = weights.squeeze(-1)
        if weights.dim() != 2:
            raise ValueError(f"region_weights must be [B,R] or [B,R,1], got {tuple(weights.shape)}")
        return weights

    @staticmethod
    def _masked_soft_normalize(
        weights: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if weights is None:
            return None

        weights = weights.clamp_min(0.0)
        if padding_mask is not None:
            if padding_mask.dim() != 2:
                raise ValueError(f"padding_mask must be [B,R], got {tuple(padding_mask.shape)}")
            weights = weights.masked_fill(padding_mask, 0.0)

        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return weights / denom

    @staticmethod
    def _default_region_weights_like(region_deep_features: torch.Tensor) -> torch.Tensor:
        if region_deep_features.dim() != 3:
            raise ValueError(f"region_deep_features must be [B,R,D], got {tuple(region_deep_features.shape)}")
        batch_size, num_regions, _ = region_deep_features.shape
        return torch.ones(batch_size, num_regions, device=region_deep_features.device, dtype=region_deep_features.dtype)

    def _build_region_type_embedding(
        self,
        batch_size: int,
        num_regions: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.region_type_embed is None:
            return None
        region_ids = torch.arange(num_regions, device=device).unsqueeze(0).expand(batch_size, num_regions)
        return self.region_type_embed(region_ids)

    def _encode_global(
        self,
        global_deep_feature: torch.Tensor,
        handcrafted_global_feature: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.global_hand_embed is not None:
            if handcrafted_global_feature is None:
                hand = global_deep_feature.new_zeros(
                    global_deep_feature.size(0),
                    self.hand_global_embed_dim,
                )
            else:
                hand = self.global_hand_embed(handcrafted_global_feature)
            return torch.cat([global_deep_feature, hand], dim=-1)
        return global_deep_feature

    def _encode_regions(
        self,
        region_deep_features: Optional[torch.Tensor],
        handcrafted_region_features: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if region_deep_features is None:
            return None

        out = region_deep_features
        batch_size, num_regions, _ = out.shape

        if self.region_hand_embed is not None:
            if handcrafted_region_features is None:
                hand = out.new_zeros(batch_size, num_regions, self.hand_region_embed_dim)
            else:
                hand = self.region_hand_embed(handcrafted_region_features)
            out = torch.cat([out, hand], dim=-1)

        type_embed = self._build_region_type_embedding(batch_size, num_regions, out.device)
        if type_embed is not None:
            out = torch.cat([out, type_embed], dim=-1)

        return out

    @staticmethod
    def _append_region_weights_to_feat(
        region_feat: torch.Tensor,
        norm_region_weights: torch.Tensor,
    ) -> torch.Tensor:
        if norm_region_weights.dim() != 2:
            raise ValueError(f"norm_region_weights must be [B,R], got {tuple(norm_region_weights.shape)}")
        return torch.cat([region_feat, norm_region_weights.unsqueeze(-1)], dim=-1)

    def region_handcrafted_logit(self, handcrafted_region_features: torch.Tensor) -> torch.Tensor:
        """
        仅基于区域手工特征计算 hand-only logit（分析用，不是主预测路径）。
        输入: [B, R, handcrafted_region_dim]
        输出: [B, R]
        """
        if self.region_hand_embed is None or self.region_hand_only_head is None:
            raise RuntimeError("region_hand_embed / region_hand_only_head is not enabled.")
        if handcrafted_region_features.dim() != 3:
            raise ValueError(
                f"handcrafted_region_features must be [B,R,D], got {tuple(handcrafted_region_features.shape)}"
            )

        hand = self.region_hand_embed(handcrafted_region_features)
        batch_size, num_regions, _ = hand.shape
        type_embed = self._build_region_type_embedding(batch_size, num_regions, hand.device)
        if type_embed is not None:
            hand = torch.cat([hand, type_embed], dim=-1)

        return self.region_hand_only_head(hand).squeeze(-1)

    def region_logit_from_fixed_deep(
        self,
        fixed_region_deep_features: torch.Tensor,
        region_weights: Optional[torch.Tensor] = None,
        handcrafted_region_features: Optional[torch.Tensor] = None,
        region_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        用真实主路径计算区域 logit：
        fixed deep region + handcrafted region + normalized region weight (+ type embedding) -> region_score_head

        输入:
            fixed_region_deep_features: [B, R, region_feature_dim]
            region_weights: [B, R] or [B, R, 1] or None
            handcrafted_region_features: [B, R, handcrafted_region_dim] or None
            region_padding_mask: [B, R] or None
        输出:
            [B, R]
        """
        if fixed_region_deep_features.dim() != 3:
            raise ValueError(
                f"fixed_region_deep_features must be [B,R,D], got {tuple(fixed_region_deep_features.shape)}"
            )

        feat = self._encode_regions(fixed_region_deep_features, handcrafted_region_features)
        if feat is None:
            raise RuntimeError("Failed to encode region features.")

        region_weights = self._prepare_region_weights(region_weights)
        if region_weights is None:
            region_weights = self._default_region_weights_like(fixed_region_deep_features)

        norm_region_weights = self._masked_soft_normalize(region_weights, region_padding_mask)
        if norm_region_weights is None:
            raise RuntimeError("Failed to normalize region weights.")

        score_input = self._append_region_weights_to_feat(feat, norm_region_weights)
        return self.region_score_head(score_input).squeeze(-1)

    def forward(
        self,
        global_deep_feature: torch.Tensor,
        region_deep_features: Optional[torch.Tensor] = None,
        region_weights: Optional[torch.Tensor] = None,
        handcrafted_global_feature: Optional[torch.Tensor] = None,
        handcrafted_region_features: Optional[torch.Tensor] = None,
        region_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        global_feat = self._encode_global(global_deep_feature, handcrafted_global_feature)
        global_logit = self.global_score_head(global_feat).squeeze(-1)
        global_score = torch.sigmoid(global_logit)

        if region_deep_features is None:
            return {
                "global_logit": global_logit,
                "global_score": global_score,
                "final_logit": global_logit,
                "final_score": global_score,
                "region_logits": None,
                "region_scores": None,
                "region_logits_raw": None,
                "region_scores_raw": None,
                "region_valid_mask": None,
                "region_weights_normalized": None,
                "prediction_region_weights_normalized": None,
                "region_logit_contributions": None,
                "region_agg_logit": None,
                "region_agg_score": None,
                "fusion_gate": None,
            }

        region_feat = self._encode_regions(region_deep_features, handcrafted_region_features)
        if region_feat is None:
            return {
                "global_logit": global_logit,
                "global_score": global_score,
                "final_logit": global_logit,
                "final_score": global_score,
                "region_logits": None,
                "region_scores": None,
                "region_logits_raw": None,
                "region_scores_raw": None,
                "region_valid_mask": None,
                "region_weights_normalized": None,
                "prediction_region_weights_normalized": None,
                "region_logit_contributions": None,
                "region_agg_logit": None,
                "region_agg_score": None,
                "fusion_gate": None,
            }

        region_weights = self._prepare_region_weights(region_weights)
        if region_weights is None:
            region_weights = self._default_region_weights_like(region_deep_features)

        norm_region_weights = self._masked_soft_normalize(region_weights, region_padding_mask)
        if norm_region_weights is None:
            return {
                "global_logit": global_logit,
                "global_score": global_score,
                "final_logit": global_logit,
                "final_score": global_score,
                "region_logits": None,
                "region_scores": None,
                "region_logits_raw": None,
                "region_scores_raw": None,
                "region_valid_mask": None,
                "region_weights_normalized": None,
                "prediction_region_weights_normalized": None,
                "region_logit_contributions": None,
                "region_agg_logit": None,
                "region_agg_score": None,
                "fusion_gate": None,
            }

        # 关键改动：区域子评分头显式吃到区域注意力权重
        region_score_input = self._append_region_weights_to_feat(region_feat, norm_region_weights)
        region_logits_raw = self.region_score_head(region_score_input).squeeze(-1)
        region_scores_raw = torch.sigmoid(region_logits_raw)

        region_logits_safe = region_logits_raw
        if region_padding_mask is not None:
            region_logits_safe = region_logits_safe.masked_fill(region_padding_mask, 0.0)

        region_logits = region_logits_raw
        region_scores = region_scores_raw
        if region_padding_mask is not None:
            region_valid_mask = ~region_padding_mask
            region_logits = region_logits.masked_fill(region_padding_mask, float("nan"))
            region_scores = region_scores.masked_fill(region_padding_mask, float("nan"))
        else:
            region_valid_mask = torch.ones_like(region_logits_raw, dtype=torch.bool)

        valid_region_exists = torch.ones(
            global_logit.shape[0],
            dtype=torch.bool,
            device=global_logit.device,
        )
        if region_padding_mask is not None:
            valid_region_exists = (~region_padding_mask).any(dim=1)

        weighted_region_logit = torch.sum(norm_region_weights * region_logits_safe, dim=1, keepdim=True)
        region_agg_logit = self.region_agg_head(weighted_region_logit).squeeze(-1)
        region_agg_logit = torch.where(valid_region_exists, region_agg_logit, torch.zeros_like(region_agg_logit))
        region_agg_score = torch.sigmoid(region_agg_logit)

        fusion_pair = torch.stack([global_logit, region_agg_logit], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(fusion_pair)).squeeze(-1)
        gate = torch.where(valid_region_exists, gate, torch.zeros_like(gate))

        mixed_logit = (1.0 - gate) * global_logit + gate * region_agg_logit
        mixed_logit = torch.where(valid_region_exists, mixed_logit, global_logit)

        final_pair = torch.stack([global_logit, mixed_logit], dim=-1)
        final_logit = self.final_fusion(final_pair).squeeze(-1)
        final_logit = torch.where(valid_region_exists, final_logit, global_logit)
        final_score = torch.sigmoid(final_logit)

        # 调试量：不是最终解释依据
        region_logit_contributions = norm_region_weights * region_logits_safe
        if region_padding_mask is not None:
            region_logit_contributions = region_logit_contributions.masked_fill(region_padding_mask, 0.0)

        return {
            "global_logit": global_logit,
            "global_score": global_score,
            "final_logit": final_logit,
            "final_score": final_score,
            "region_logits": region_logits,
            "region_scores": region_scores,
            "region_logits_raw": region_logits_raw,
            "region_scores_raw": region_scores_raw,
            "region_valid_mask": region_valid_mask,
            "region_weights_normalized": norm_region_weights,
            "prediction_region_weights_normalized": norm_region_weights,
            "region_logit_contributions": region_logit_contributions,
            "region_agg_logit": region_agg_logit,
            "region_agg_score": region_agg_score,
            "fusion_gate": gate,
        }
