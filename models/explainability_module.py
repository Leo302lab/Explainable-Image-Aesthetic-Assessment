from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None

from utils.config import denormalize_score


META_FEATURES = {"valid", "area_ratio"}


@dataclass
class RegionExplanation:
    region_index: int
    region_name: str
    region_score_10: float
    region_weight: float
    region_contribution: float
    positive_features: List[Tuple[str, float]]
    negative_features: List[Tuple[str, float]]
    sentence: str


class _FullModelFinalScoreWrapper(nn.Module):
    """
    固定 deep features / patch tokens / region padding mask，
    只让 handcrafted features 作为输入，解释它们对 final_logit / final_score 的影响。
    """

    def __init__(
        self,
        full_model: nn.Module,
        fixed_global_deep: torch.Tensor,
        fixed_region_deep: Optional[torch.Tensor],
        fixed_patch_tokens: torch.Tensor,
        fixed_region_padding_mask: Optional[torch.Tensor],
        global_dim: int,
        num_regions: int,
        region_dim: int,
        use_final_logit: bool = True,
    ) -> None:
        super().__init__()
        self.full_model = full_model
        self.register_buffer("fixed_global_deep", fixed_global_deep.detach().clone())
        self.register_buffer(
            "fixed_region_deep",
            torch.empty(0) if fixed_region_deep is None else fixed_region_deep.detach().clone(),
        )
        self.register_buffer("fixed_patch_tokens", fixed_patch_tokens.detach().clone())
        self.register_buffer(
            "fixed_region_padding_mask",
            torch.empty(0, dtype=torch.bool)
            if fixed_region_padding_mask is None
            else fixed_region_padding_mask.detach().clone(),
        )
        self.global_dim = int(global_dim)
        self.num_regions = int(num_regions)
        self.region_dim = int(region_dim)
        self.use_final_logit = bool(use_final_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected handcrafted input shape [B, D], got {tuple(x.shape)}")

        device = self.fixed_global_deep.device
        x = x.to(device)

        handcrafted_global = x[:, : self.global_dim]
        handcrafted_region = x[:, self.global_dim :].reshape(-1, self.num_regions, self.region_dim)

        batch_size = x.size(0)
        fixed_region_deep = None if self.fixed_region_deep.numel() == 0 else self.fixed_region_deep.expand(batch_size, -1, -1)
        fixed_region_padding_mask = (
            None if self.fixed_region_padding_mask.numel() == 0 else self.fixed_region_padding_mask.expand(batch_size, -1)
        )

        out = self.full_model.forward_from_intermediate_features(
            global_deep_feature=self.fixed_global_deep.expand(batch_size, -1),
            region_deep_features=fixed_region_deep,
            patch_tokens=self.fixed_patch_tokens.expand(batch_size, -1, -1),
            handcrafted_global=handcrafted_global,
            handcrafted_region=handcrafted_region,
            region_padding_mask=fixed_region_padding_mask,
            apply_region_dropout=False,
        )
        key = "final_logit" if self.use_final_logit else "final_score"
        return out[key].unsqueeze(-1)


class ExplainabilityModule:
    """
    对齐原始设计：
    1) SHAP 值计算各 handcrafted 特征对全局 final_score 的贡献；
    2) 区域贡献改为基于 final_score 的显式消融贡献；
    3) 局部区域解释从“全局 final_score 的 SHAP 结果”中按区域切片汇总，而不是解释 hand-only 辅助头。
    """

    def __init__(
        self,
        region_names: Sequence[str],
        region_feature_names: Sequence[str],
        global_feature_names: Optional[Sequence[str]] = None,
        score_range: Tuple[float, float] = (1.0, 10.0),
    ) -> None:
        self.region_names = list(region_names)
        self.region_feature_names = list(region_feature_names)
        self.global_feature_names = None if global_feature_names is None else list(global_feature_names)
        self.score_range = score_range

    @staticmethod
    def _top_signed_features(
        values: np.ndarray,
        feature_names: Sequence[str],
        top_k: int = 3,
        exclude_meta: bool = True,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        names_vals = [(str(name), float(val)) for name, val in zip(feature_names, values)]
        if exclude_meta:
            names_vals = [(n, v) for n, v in names_vals if not any(n.endswith(f"_{m}") or n == m for m in META_FEATURES)]
        pos = sorted([x for x in names_vals if x[1] > 0], key=lambda x: x[1], reverse=True)[:top_k]
        neg = sorted([x for x in names_vals if x[1] < 0], key=lambda x: x[1])[:top_k]
        return pos, neg

    def _resolve_global_feature_names(self, dim: int) -> List[str]:
        if self.global_feature_names is not None and len(self.global_feature_names) == dim:
            return list(self.global_feature_names)
        return [f"global_feat_{i}" for i in range(dim)]

    def _resolve_region_feature_names(self, dim: int) -> List[str]:
        if len(self.region_feature_names) == dim:
            return list(self.region_feature_names)
        return [f"region_feat_{i}" for i in range(dim)]

    def _build_flat_feature_names(self, global_dim: int, region_dim: int) -> List[str]:
        global_names = [f"global_{n}" for n in self._resolve_global_feature_names(global_dim)]
        region_names = self._resolve_region_feature_names(region_dim)
        flat = list(global_names)
        for ridx, rname in enumerate(self.region_names):
            prefix = rname if rname else f"region_{ridx}"
            for feat_name in region_names:
                flat.append(f"{prefix}_{feat_name}")
        return flat

    @staticmethod
    def _flatten_handcrafted(global_feat: torch.Tensor, region_feat: torch.Tensor) -> torch.Tensor:
        if global_feat.dim() != 2:
            raise ValueError(f"global_feat must be [B, G], got {tuple(global_feat.shape)}")
        if region_feat.dim() != 3:
            raise ValueError(f"region_feat must be [B, R, D], got {tuple(region_feat.shape)}")
        return torch.cat([global_feat, region_feat.flatten(start_dim=1)], dim=1)

    @staticmethod
    def _ensure_numpy_shap_output(shap_values) -> np.ndarray:
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return np.asarray(shap_values)

    @staticmethod
    def _grad_times_input(wrapper: nn.Module, x: torch.Tensor) -> np.ndarray:
        x = x.clone().detach().requires_grad_(True)
        y = wrapper(x).sum()
        y.backward()
        return (x.grad.detach().cpu().numpy() * x.detach().cpu().numpy()).astype(np.float32)

    @staticmethod
    def build_contribution_map(region_soft_masks: torch.Tensor, region_contributions: torch.Tensor) -> torch.Tensor:
        return torch.sum(region_soft_masks * region_contributions.unsqueeze(-1).unsqueeze(-1), dim=1)

    def _build_wrapper(
        self,
        full_model: nn.Module,
        outputs: Dict[str, torch.Tensor],
        handcrafted_global: torch.Tensor,
        handcrafted_region: torch.Tensor,
        use_final_logit: bool = True,
    ) -> _FullModelFinalScoreWrapper:
        return _FullModelFinalScoreWrapper(
            full_model=full_model,
            fixed_global_deep=outputs["deep_global_feature"].detach(),
            fixed_region_deep=outputs.get("deep_region_features"),
            fixed_patch_tokens=outputs["patch_tokens"].detach(),
            fixed_region_padding_mask=outputs.get("region_padding_mask"),
            global_dim=handcrafted_global.size(-1),
            num_regions=handcrafted_region.size(1),
            region_dim=handcrafted_region.size(2),
            use_final_logit=use_final_logit,
        ).eval()

    def _compute_global_final_shap_values(
        self,
        full_model: nn.Module,
        outputs: Dict[str, torch.Tensor],
        background_global_features: torch.Tensor,
        background_region_features: torch.Tensor,
        sample_global_features: torch.Tensor,
        sample_region_features: torch.Tensor,
        use_final_logit: bool = True,
    ) -> np.ndarray:
        wrapper = self._build_wrapper(
            full_model=full_model,
            outputs=outputs,
            handcrafted_global=sample_global_features,
            handcrafted_region=sample_region_features,
            use_final_logit=use_final_logit,
        )

        device = outputs["deep_global_feature"].device
        bg = self._flatten_handcrafted(
            background_global_features.detach().to(device),
            background_region_features.detach().to(device),
        )
        x = self._flatten_handcrafted(
            sample_global_features.detach().to(device),
            sample_region_features.detach().to(device),
        )

        if shap is None:
            return self._grad_times_input(wrapper, x)

        try:
            explainer = shap.GradientExplainer(wrapper, bg)
            shap_values = explainer.shap_values(x)
            return self._ensure_numpy_shap_output(shap_values)
        except Exception:
            return self._grad_times_input(wrapper, x)

    def compute_region_counterfactual_contributions(
        self,
        full_model: nn.Module,
        outputs: Dict[str, torch.Tensor],
        background_global_features: torch.Tensor,
        background_region_features: torch.Tensor,
        handcrafted_global: torch.Tensor,
        handcrafted_region: torch.Tensor,
    ) -> torch.Tensor:
        """
        基于 final_score_10 的显式消融贡献：
        固定当前样本 deep/context，仅把某个区域的核心 handcrafted 特征替换成背景均值，
        重新计算 final_score_10，base - ablated 即该区域对整图最终评分的贡献。
        """
        if handcrafted_global.dim() != 2 or handcrafted_global.size(0) != 1:
            raise ValueError(f"Expected handcrafted_global [1,G], got {tuple(handcrafted_global.shape)}")
        if handcrafted_region.dim() != 3 or handcrafted_region.size(0) != 1:
            raise ValueError(f"Expected handcrafted_region [1,R,D], got {tuple(handcrafted_region.shape)}")

        device = outputs["deep_global_feature"].device
        handcrafted_global = handcrafted_global.detach().to(device)
        handcrafted_region = handcrafted_region.detach().to(device)
        background_region_features = background_region_features.detach().to(device)

        base_score_10 = outputs.get("final_score_10")
        if base_score_10 is None:
            base_score_10 = denormalize_score(outputs["final_score"], self.score_range)
        base_score_10 = float(base_score_10[0].item())

        region_dim = handcrafted_region.size(-1)
        region_feature_names = self._resolve_region_feature_names(region_dim)
        core_indices = [i for i, n in enumerate(region_feature_names) if n not in META_FEATURES]
        if not core_indices:
            core_indices = list(range(region_dim))

        region_mean = background_region_features.mean(dim=0, keepdim=False)  # [R,D]
        num_regions = handcrafted_region.size(1)
        contributions = []

        for ridx in range(num_regions):
            ablated_region = handcrafted_region.clone()
            ablated_region[:, ridx, core_indices] = region_mean[ridx, core_indices]

            ablated_out = full_model.forward_from_intermediate_features(
                global_deep_feature=outputs["deep_global_feature"],
                region_deep_features=outputs.get("deep_region_features"),
                patch_tokens=outputs["patch_tokens"],
                handcrafted_global=handcrafted_global,
                handcrafted_region=ablated_region,
                region_padding_mask=outputs.get("region_padding_mask"),
                image_region_soft_masks=outputs.get("image_region_soft_masks"),
                region_soft_masks=outputs.get("region_soft_masks"),
                apply_region_dropout=False,
            )
            ablated_score_10 = float(ablated_out["final_score_10"][0].item())
            contributions.append(base_score_10 - ablated_score_10)

        contributions_t = torch.tensor(contributions, dtype=torch.float32, device=device)
        region_padding_mask = outputs.get("region_padding_mask")
        if region_padding_mask is not None:
            contributions_t = contributions_t.masked_fill(region_padding_mask[0], 0.0)
        return contributions_t

    def explain_global_final_score(
        self,
        outputs: Dict[str, torch.Tensor],
        full_model: nn.Module,
        background_global_features: torch.Tensor,
        background_region_features: torch.Tensor,
        handcrafted_global: Optional[torch.Tensor] = None,
        handcrafted_region: Optional[torch.Tensor] = None,
        top_k_features: int = 6,
        top_k_regions: int = 2,
    ) -> Dict[str, object]:
        if handcrafted_global is None:
            handcrafted_global = outputs.get("raw_handcrafted_global")
        if handcrafted_region is None:
            handcrafted_region = outputs.get("raw_handcrafted_region")

        if handcrafted_global is None or handcrafted_region is None:
            raise ValueError("handcrafted_global and handcrafted_region are required for global SHAP.")
        if handcrafted_global.dim() != 2 or handcrafted_global.size(0) != 1:
            raise ValueError(f"Expected handcrafted_global [1, G], got {tuple(handcrafted_global.shape)}")
        if handcrafted_region.dim() != 3 or handcrafted_region.size(0) != 1:
            raise ValueError(f"Expected handcrafted_region [1, R, D], got {tuple(handcrafted_region.shape)}")

        shap_values = self._compute_global_final_shap_values(
            full_model=full_model,
            outputs=outputs,
            background_global_features=background_global_features,
            background_region_features=background_region_features,
            sample_global_features=handcrafted_global,
            sample_region_features=handcrafted_region,
            use_final_logit=True,
        )[0]

        flat_feature_names = self._build_flat_feature_names(handcrafted_global.size(-1), handcrafted_region.size(-1))
        positive_features, negative_features = self._top_signed_features(
            shap_values, flat_feature_names, top_k=top_k_features, exclude_meta=True
        )

        region_contributions = self.compute_region_counterfactual_contributions(
            full_model=full_model,
            outputs=outputs,
            background_global_features=background_global_features,
            background_region_features=background_region_features,
            handcrafted_global=handcrafted_global,
            handcrafted_region=handcrafted_region,
        )

        region_padding_mask = outputs.get("region_padding_mask")
        valid_mask = torch.ones_like(region_contributions, dtype=torch.bool)
        if region_padding_mask is not None:
            valid_mask = ~region_padding_mask[0]
        valid_indices = torch.where(valid_mask)[0].tolist()
        ranked = sorted(valid_indices, key=lambda i: abs(float(region_contributions[i])), reverse=True)
        selected_regions = ranked[:top_k_regions]

        final_score_10 = outputs.get("final_score_10")
        if final_score_10 is None:
            final_score_10 = denormalize_score(outputs["final_score"], self.score_range)

        summary = f"整图最终美学评分为 {float(final_score_10[0].item()):.2f}/10。"
        if positive_features:
            summary += " 主要正向特征：" + "、".join(name for name, _ in positive_features[:3]) + "。"
        if negative_features:
            summary += " 主要负向特征：" + "、".join(name for name, _ in negative_features[:3]) + "。"
        if selected_regions:
            region_text = []
            for idx in selected_regions:
                rname = self.region_names[idx] if idx < len(self.region_names) else f"region_{idx}"
                region_text.append(f"{rname}({float(region_contributions[idx]):+.3f})")
            summary += " 关键区域贡献：" + "、".join(region_text) + "。"

        return {
            "final_score_10": float(final_score_10[0].item()),
            "positive_features": positive_features,
            "negative_features": negative_features,
            "selected_regions": selected_regions,
            "region_contributions": region_contributions.detach().cpu(),
            "flat_feature_names": flat_feature_names,
            "shap_values": shap_values,
            "summary": summary,
        }

    def explain_region_local(
        self,
        outputs: Dict[str, torch.Tensor],
        global_shap_values: np.ndarray,
        region_idx: int,
        handcrafted_region: Optional[torch.Tensor] = None,
        top_k_features: int = 3,
        region_contributions: Optional[torch.Tensor] = None,
    ) -> RegionExplanation:
        if handcrafted_region is None:
            handcrafted_region = outputs.get("raw_handcrafted_region")
        if handcrafted_region is None:
            raise ValueError("handcrafted_region is required for local region explanation.")
        if handcrafted_region.dim() != 3 or handcrafted_region.size(0) != 1:
            raise ValueError(f"Expected handcrafted_region [1, R, D], got {tuple(handcrafted_region.shape)}")

        region_dim = handcrafted_region.size(-1)
        region_feature_names = self._resolve_region_feature_names(region_dim)
        global_dim = int(outputs.get("raw_handcrafted_global").size(-1)) if outputs.get("raw_handcrafted_global") is not None else 0

        start = global_dim + region_idx * region_dim
        end = start + region_dim
        local_values = np.asarray(global_shap_values[start:end], dtype=np.float32)
        positive_features, negative_features = self._top_signed_features(
            local_values, region_feature_names, top_k=top_k_features, exclude_meta=True
        )

        region_scores_10 = outputs.get("region_scores_10")
        if region_scores_10 is None and outputs.get("region_scores") is not None:
            region_scores_10 = denormalize_score(outputs["region_scores"], self.score_range)

        region_weights = outputs.get("prediction_region_weights_normalized")
        if region_weights is None:
            region_weights = outputs.get("region_weights")

        region_name = self.region_names[region_idx] if region_idx < len(self.region_names) else f"region_{region_idx}"
        r_score = 0.0 if region_scores_10 is None else float(region_scores_10[0, region_idx].item())
        r_weight = 0.0 if region_weights is None else float(region_weights[0, region_idx].item())
        r_contrib = 0.0 if region_contributions is None else float(region_contributions[region_idx].item())

        if r_contrib >= 0:
            sentence = (
                f"{region_name}区域对整体审美有正向作用（区域分 {r_score:.2f}），"
                f"主要受 {', '.join(name for name, _ in positive_features) or '稳定特征'} 的正向影响。"
            )
        else:
            sentence = (
                f"{region_name}区域对整体审美有负向作用（区域分 {r_score:.2f}），"
                f"主要问题集中在 {', '.join(name for name, _ in negative_features) or '关键负向特征'}。"
            )

        return RegionExplanation(
            region_index=region_idx,
            region_name=region_name,
            region_score_10=r_score,
            region_weight=r_weight,
            region_contribution=r_contrib,
            positive_features=positive_features,
            negative_features=negative_features,
            sentence=sentence,
        )

    def explain_single(
        self,
        outputs: Dict[str, torch.Tensor],
        full_model: nn.Module,
        background_global_features: torch.Tensor,
        background_region_features: torch.Tensor,
        handcrafted_global: Optional[torch.Tensor] = None,
        handcrafted_region: Optional[torch.Tensor] = None,
        top_k_regions: int = 2,
        top_k_features: int = 3,
    ) -> Dict[str, object]:
        global_result = self.explain_global_final_score(
            outputs=outputs,
            full_model=full_model,
            background_global_features=background_global_features,
            background_region_features=background_region_features,
            handcrafted_global=handcrafted_global,
            handcrafted_region=handcrafted_region,
            top_k_features=max(4, top_k_features),
            top_k_regions=top_k_regions,
        )

        region_explanations: List[RegionExplanation] = []
        for region_idx in global_result["selected_regions"]:
            region_explanations.append(
                self.explain_region_local(
                    outputs=outputs,
                    global_shap_values=global_result["shap_values"],
                    region_idx=int(region_idx),
                    handcrafted_region=handcrafted_region if handcrafted_region is not None else outputs.get("raw_handcrafted_region"),
                    top_k_features=top_k_features,
                    region_contributions=global_result["region_contributions"],
                )
            )

        contribution_map = None
        masks = outputs.get("image_region_soft_masks")
        if masks is None:
            masks = outputs.get("region_soft_masks")
        if masks is not None and global_result["region_contributions"] is not None:
            contribution_map = self.build_contribution_map(
                masks,
                global_result["region_contributions"].unsqueeze(0).to(masks.device),
            )

        summary = global_result["summary"]
        if region_explanations:
            summary += " 关键区域解释：" + " ".join(exp.sentence for exp in region_explanations)

        return {
            "final_score_10": global_result["final_score_10"],
            "global_positive_features": global_result["positive_features"],
            "global_negative_features": global_result["negative_features"],
            "region_explanations": region_explanations,
            "region_contributions": global_result["region_contributions"],
            "contribution_map": contribution_map,
            "global_shap_values": global_result["shap_values"],
            "global_feature_names": global_result["flat_feature_names"],
            "summary": summary,
        }
