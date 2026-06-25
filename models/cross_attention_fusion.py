from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Region-query / global-patch-key-value cross attention.

    保留当前主干结构，只补解释诊断所需输出：
    1. head_importance：用于更稳定地聚合多头区域注意力
    2. context_region_features_hidden 等隐藏态继续透传，便于解释阶段复算
    """

    def __init__(
        self,
        query_dim: Optional[int] = None,
        kv_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.2,
        region_dim: Optional[int] = None,
        global_dim: Optional[int] = None,
        fusion_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if query_dim is None:
            query_dim = region_dim
        if kv_dim is None:
            kv_dim = global_dim
        if hidden_dim is None:
            hidden_dim = fusion_dim

        if query_dim is None:
            raise ValueError("CrossAttentionFusion requires query_dim (or legacy region_dim).")
        if kv_dim is None:
            raise ValueError("CrossAttentionFusion requires kv_dim (or legacy global_dim).")
        if hidden_dim is None:
            raise ValueError("CrossAttentionFusion requires hidden_dim (or legacy fusion_dim).")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")

        if output_dim is None:
            output_dim = hidden_dim

        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = int(num_heads)

        self.query_in_proj = nn.Linear(query_dim, hidden_dim) if query_dim != hidden_dim else nn.Identity()
        self.kv_in_proj = nn.Linear(kv_dim, hidden_dim) if kv_dim != hidden_dim else nn.Identity()
        self.global_in_proj = nn.Linear(kv_dim, hidden_dim) if kv_dim != hidden_dim else nn.Identity()

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.region_ffn_norm = nn.LayerNorm(hidden_dim)
        self.region_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.global_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.region_out_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        self.fused_out_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        self.pooled_out_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
        if mask is None:
            return torch.softmax(logits, dim=dim)

        logits = logits.masked_fill(mask, float("-inf"))
        all_masked = mask.all(dim=dim, keepdim=True)
        safe_logits = torch.where(all_masked, torch.zeros_like(logits), logits)
        weights = torch.softmax(safe_logits, dim=dim)
        weights = torch.where(all_masked, torch.zeros_like(weights), weights)
        return weights

    @staticmethod
    def _compute_head_importance(
        attn_weights: torch.Tensor,
        region_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        基于每个 head 在 patch 维度上的注意力聚焦程度（平均最大注意力）估计 head importance。
        attn_weights: [B, H, R, P]
        return: [B, H]
        """
        if attn_weights.dim() != 4:
            raise ValueError(f"Expected attn_weights [B,H,R,P], got {tuple(attn_weights.shape)}")

        head_focus = attn_weights.max(dim=-1).values  # [B,H,R]
        if region_padding_mask is not None:
            region_mask = region_padding_mask.unsqueeze(1)  # [B,1,R]
            head_focus = head_focus.masked_fill(region_mask, 0.0)
            valid_count = (~region_mask).sum(dim=-1).clamp_min(1)
            head_focus = head_focus.sum(dim=-1) / valid_count
        else:
            head_focus = head_focus.mean(dim=-1)

        head_importance = torch.softmax(head_focus, dim=1)
        return head_importance

    def forward(
        self,
        global_feature: torch.Tensor,
        global_patch_tokens: torch.Tensor,
        region_features: torch.Tensor,
        region_padding_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if global_feature.ndim != 2:
            raise ValueError("global_feature must have shape [B, kv_dim].")
        if global_patch_tokens.ndim != 3:
            raise ValueError("global_patch_tokens must have shape [B, P, kv_dim].")
        if region_features.ndim != 3:
            raise ValueError("region_features must have shape [B, R, query_dim].")

        batch_size = region_features.size(0)
        if global_feature.size(0) != batch_size or global_patch_tokens.size(0) != batch_size:
            raise ValueError("Batch size mismatch among global_feature, global_patch_tokens, and region_features.")

        if global_feature.size(-1) != self.kv_dim:
            raise ValueError(f"global_feature last dim must be kv_dim={self.kv_dim}, got {global_feature.size(-1)}.")
        if global_patch_tokens.size(-1) != self.kv_dim:
            raise ValueError(
                f"global_patch_tokens last dim must be kv_dim={self.kv_dim}, got {global_patch_tokens.size(-1)}."
            )
        if region_features.size(-1) != self.query_dim:
            raise ValueError(
                f"region_features last dim must be query_dim={self.query_dim}, got {region_features.size(-1)}."
            )

        if region_padding_mask is not None:
            if region_padding_mask.shape != region_features.shape[:2]:
                raise ValueError("region_padding_mask must have shape [B, R] matching region_features.")
            region_padding_mask = region_padding_mask.bool()

        if patch_padding_mask is not None:
            if patch_padding_mask.shape != global_patch_tokens.shape[:2]:
                raise ValueError("patch_padding_mask must have shape [B, P] matching global_patch_tokens.")
            patch_padding_mask = patch_padding_mask.bool()

        query_hidden = self.query_in_proj(region_features)
        kv_hidden = self.kv_in_proj(global_patch_tokens)
        global_hidden = self.global_in_proj(global_feature)

        query = self.query_norm(query_hidden)
        key_value = self.kv_norm(kv_hidden)

        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=patch_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )  # [B,H,R,P]

        context_regions_hidden = query_hidden + attn_out
        context_regions_hidden = context_regions_hidden + self.region_ffn(self.region_ffn_norm(context_regions_hidden))

        if region_padding_mask is not None:
            context_regions_hidden = context_regions_hidden.masked_fill(region_padding_mask.unsqueeze(-1), 0.0)

        global_hidden_expand = global_hidden.unsqueeze(1).expand_as(context_regions_hidden)
        importance_input = torch.cat([context_regions_hidden, global_hidden_expand], dim=-1)
        importance_logits = self.importance_head(importance_input).squeeze(-1)
        region_weights = self._masked_softmax(importance_logits, region_padding_mask, dim=1)

        pooled_regions_hidden = torch.sum(region_weights.unsqueeze(-1) * context_regions_hidden, dim=1)
        fused_hidden = self.global_fuse(torch.cat([global_hidden, pooled_regions_hidden], dim=-1))

        head_importance = self._compute_head_importance(attn_weights, region_padding_mask)

        context_regions = self.region_out_proj(context_regions_hidden)
        pooled_regions = self.pooled_out_proj(pooled_regions_hidden)
        fused_feature = self.fused_out_proj(fused_hidden)

        return {
            "fused_feature": fused_feature,
            "fused_feature_hidden": fused_hidden,
            "context_region_features": context_regions,
            "context_region_features_hidden": context_regions_hidden,
            "region_importance": region_weights,
            "region_importance_logits": importance_logits,
            "head_importance": head_importance,
            "cross_attn_weights": attn_weights,
            "pooled_region_feature": pooled_regions,
            "pooled_region_feature_hidden": pooled_regions_hidden,
        }
