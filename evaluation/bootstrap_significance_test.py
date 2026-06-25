#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired bootstrap significance test for SSG-CAF vs baseline models.

This script compares one proposed model with several baselines using paired
bootstrap resampling over the held-out test images.

Expected prediction files
-------------------------
Default pred_dir mode expects one CSV per model and seed, for example:

    SSG_CAF_seed42.csv
    SSG_CAF_seed123.csv
    SSG_CAF_seed2026.csv
    ResNet50_MLP_seed42.csv
    ViT_Base_seed42.csv
    Spatial_GCN_seed42.csv
    TAVAR_seed42.csv
    MUSIQ_style_seed42.csv

Each CSV should contain at least three columns:

    image_id, label_raw, pred_raw

Column names are detected automatically. Supported alternatives:

    image id: image_id, img_id, id, filename, file_name, path
    target  : label_raw, target_raw, score, target, label, y_true, gt
    pred    : pred_raw, prediction_raw, pred_score, pred, y_pred, prediction

Important
---------
MSE and MAE are scale-dependent. To reproduce the paper table, make sure that
label and prediction are on the same scale as the reported metrics, usually the
raw AVA score scale rather than the normalised 0-1 scale.

Example
-------
python bootstrap_significance_test.py \
    --pred_dir /root/autodl-tmp/pythonProject2/checkpoints/predictions \
    --ours SSG_CAF \
    --baselines ResNet50_MLP ViT_Base Spatial_GCN TAVAR MUSIQ_style \
    --seeds 42 123 2026 \
    --n_boot 5000 \
    --out_csv bootstrap_significance_results.csv

Definitions
-----------
For MSE and MAE:
    delta = baseline_metric - ours_metric
For PLCC and SRCC:
    delta = ours_metric - baseline_metric

Positive delta always means that SSG-CAF performs better.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ============================================================
# 直接在 IDE 运行时使用的默认配置。
# 如果你想在 terminal 用命令行参数覆盖，依旧可以：
#   python bootstrap_significance_test.py --pred_dir ...
# 否则脚本会使用下面这组参数直接开始运行。
# ============================================================
_DEFAULT_CONFIG = {
    # 预测 CSV 所在目录，要求文件名形如 {model}_seed{seed}.csv
    "pred_dir": "/root/autodl-tmp/pythonProject2/checkpoints/predictions",

    # 对比模型名：我们的方法 vs 基线
    "ours": "SSG_CAF_full",
    "baselines": ["MUSIQ_style"],
    "seeds": [42],

    # 指标与 bootstrap 次数
    "metrics": ["mse", "mae", "plcc", "srcc"],
    "n_boot": 5000,          # 跑 5000 次；如赶时间可先临时改成 200 验证
    "bootstrap_seed": 2026,

    # 输出文件（相对脚本目录）
    "out_csv": "bootstrap_significance_results.csv",
    "point_csv": "bootstrap_point_metrics.csv",
    "save_deltas_npz": None,
    "file_pattern": "{model}_seed{seed}.csv",
}


ID_COLUMNS = ["image_id", "img_id", "id", "filename", "file_name", "path", "image", "name"]
TARGET_COLUMNS = ["label_raw", "target_raw", "score_raw", "gt_raw", "target", "label", "score", "y_true", "gt"]
PRED_COLUMNS = ["pred_raw", "prediction_raw", "pred_score", "pred", "prediction", "y_pred", "output"]
MODEL_COLUMNS = ["model", "model_name", "method"]
SEED_COLUMNS = ["seed", "random_seed"]


def _args_from_default_config() -> argparse.Namespace:
    """构造一套使用 _DEFAULT_CONFIG 的 args（IDE 直接运行用）。"""
    ns = argparse.Namespace()
    ns.pred_dir = _DEFAULT_CONFIG["pred_dir"]
    ns.combined_csv = None
    ns.mapping_json = None
    ns.ours = _DEFAULT_CONFIG["ours"]
    ns.baselines = list(_DEFAULT_CONFIG["baselines"])
    ns.seeds = list(_DEFAULT_CONFIG["seeds"])
    ns.metrics = list(_DEFAULT_CONFIG["metrics"])
    ns.n_boot = int(_DEFAULT_CONFIG["n_boot"])
    ns.bootstrap_seed = int(_DEFAULT_CONFIG["bootstrap_seed"])
    ns.file_pattern = _DEFAULT_CONFIG["file_pattern"]
    ns.out_csv = _DEFAULT_CONFIG["out_csv"]
    ns.point_csv = _DEFAULT_CONFIG["point_csv"]
    ns.save_deltas_npz = _DEFAULT_CONFIG["save_deltas_npz"]
    return ns


def parse_args():
    # ① 如果是在 terminal 里带命令行参数运行：使用 argparse。
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Paired bootstrap test for IAA baselines.")

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("--pred_dir", type=str, help="Directory containing one CSV per model and seed.")
        group.add_argument("--combined_csv", type=str, help="Combined CSV with model, seed, image_id, target and prediction columns.")
        group.add_argument("--mapping_json", type=str, help="JSON mapping model names and seeds to CSV paths.")

        parser.add_argument("--ours", type=str, default=_DEFAULT_CONFIG["ours"])
        parser.add_argument("--baselines", nargs="+", default=list(_DEFAULT_CONFIG["baselines"]))
        parser.add_argument("--seeds", nargs="+", type=int, default=list(_DEFAULT_CONFIG["seeds"]))
        parser.add_argument("--metrics", nargs="+", default=list(_DEFAULT_CONFIG["metrics"]),
                            choices=["mse", "mae", "plcc", "srcc"])
        parser.add_argument("--n_boot", type=int, default=_DEFAULT_CONFIG["n_boot"])
        parser.add_argument("--bootstrap_seed", type=int, default=_DEFAULT_CONFIG["bootstrap_seed"])
        parser.add_argument("--file_pattern", type=str, default=_DEFAULT_CONFIG["file_pattern"])
        parser.add_argument("--out_csv", type=str, default=_DEFAULT_CONFIG["out_csv"])
        parser.add_argument("--point_csv", type=str, default=_DEFAULT_CONFIG["point_csv"])
        parser.add_argument("--save_deltas_npz", type=str, default=_DEFAULT_CONFIG["save_deltas_npz"])
        args = parser.parse_args()
        # 如果用户没有显式给 pred_dir/combined_csv/mapping_json，则回退到默认 pred_dir
        if not (args.pred_dir or args.combined_csv or args.mapping_json):
            args.pred_dir = _DEFAULT_CONFIG["pred_dir"]
        return args

    # ② 否则：IDE 直接运行 → 使用内置默认配置
    return _args_from_default_config()


def safe_name(name: str) -> str:
    x = str(name).strip().replace("+", "_plus_")
    x = re.sub(r"[^A-Za-z0-9]+", "_", x)
    return re.sub(r"_+", "_", x).strip("_")


def find_first_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Cannot find required columns {candidates}. Existing columns: {list(df.columns)}")
    return None


def normalise_prediction_df(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    id_col = find_first_column(df, ID_COLUMNS)
    target_col = find_first_column(df, TARGET_COLUMNS)
    pred_col = find_first_column(df, PRED_COLUMNS)

    out = pd.DataFrame({
        "image_id": df[id_col].astype(str),
        "target": pd.to_numeric(df[target_col], errors="coerce"),
        "pred": pd.to_numeric(df[pred_col], errors="coerce"),
    })
    out = out.dropna(subset=["image_id", "target", "pred"])
    out = out.drop_duplicates(subset=["image_id"], keep="first")
    if out.empty:
        raise ValueError(f"No valid rows found in {source}")
    return out


def load_single_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prediction file not found: {p}")
    return normalise_prediction_df(pd.read_csv(p), str(p))


def candidate_paths(pred_dir: Path, model: str, seed: int, pattern: str) -> List[Path]:
    s = safe_name(model)
    candidates = [
        pred_dir / pattern.format(model=model, safe_model=s, seed=seed),
        pred_dir / pattern.format(model=s, safe_model=s, seed=seed),
        pred_dir / f"{model}_seed{seed}.csv",
        pred_dir / f"{s}_seed{seed}.csv",
        pred_dir / f"{model}_predictions_seed{seed}.csv",
        pred_dir / f"{s}_predictions_seed{seed}.csv",
        pred_dir / f"seed{seed}_{model}.csv",
        pred_dir / f"seed{seed}_{s}.csv",
    ]
    unique = []
    seen = set()
    for c in candidates:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def load_from_pred_dir(pred_dir: str, model: str, seed: int, pattern: str) -> pd.DataFrame:
    d = Path(pred_dir)
    for p in candidate_paths(d, model, seed, pattern):
        if p.exists():
            return load_single_csv(str(p))
    tried = "\n".join(str(p) for p in candidate_paths(d, model, seed, pattern))
    raise FileNotFoundError(f"Cannot find CSV for model={model}, seed={seed}. Tried:\n{tried}")


def load_from_combined_csv(combined_csv: str, model: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(combined_csv)
    model_col = find_first_column(df, MODEL_COLUMNS)
    seed_col = find_first_column(df, SEED_COLUMNS)
    seed_series = pd.to_numeric(df[seed_col], errors="coerce").astype("Int64")

    sub = df[(df[model_col].astype(str) == str(model)) & (seed_series == int(seed))].copy()
    if sub.empty:
        sub = df[(df[model_col].astype(str).map(safe_name) == safe_name(model)) & (seed_series == int(seed))].copy()
    if sub.empty:
        raise ValueError(f"No rows in combined CSV for model={model}, seed={seed}")
    return normalise_prediction_df(sub, f"{combined_csv}:{model}:seed{seed}")


def load_mapping_json(mapping_json: str) -> Dict:
    with open(mapping_json, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if "ours" not in mapping or "baselines" not in mapping:
        raise ValueError("mapping_json must contain keys: ours and baselines")
    return mapping


def load_predictions(args) -> Dict[str, Dict[int, pd.DataFrame]]:
    all_models = [args.ours] + list(args.baselines)
    data: Dict[str, Dict[int, pd.DataFrame]] = {m: {} for m in all_models}

    if args.mapping_json:
        mapping = load_mapping_json(args.mapping_json)
        for seed in args.seeds:
            data[args.ours][seed] = load_single_csv(mapping["ours"][str(seed)])
        for base in args.baselines:
            for seed in args.seeds:
                data[base][seed] = load_single_csv(mapping["baselines"][base][str(seed)])
        return data

    if args.combined_csv:
        for model in all_models:
            for seed in args.seeds:
                data[model][seed] = load_from_combined_csv(args.combined_csv, model, seed)
        return data

    for model in all_models:
        for seed in args.seeds:
            data[model][seed] = load_from_pred_dir(args.pred_dir, model, seed, args.file_pattern)
    return data


def metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 2:
        return 0.0
    if metric == "mse":
        return float(np.mean((y_pred - y_true) ** 2))
    if metric == "mae":
        return float(np.mean(np.abs(y_pred - y_true)))
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    if metric == "plcc":
        v = float(np.corrcoef(y_true, y_pred)[0, 1])
        return 0.0 if np.isnan(v) else v
    if metric == "srcc":
        v = float(spearmanr(y_true, y_pred).correlation)
        return 0.0 if np.isnan(v) else v
    raise ValueError(metric)


def delta_value(ours_metric: float, base_metric: float, metric: str) -> float:
    # Positive delta always means SSG-CAF is better.
    if metric in ("mse", "mae"):
        return base_metric - ours_metric
    return ours_metric - base_metric


def align_for_comparison(data: Dict[str, Dict[int, pd.DataFrame]], ours_name: str, base_name: str, seeds: List[int]) -> Dict[int, pd.DataFrame]:
    aligned: Dict[int, pd.DataFrame] = {}
    common_all = None

    for seed in seeds:
        ours = data[ours_name][seed].rename(columns={"pred": "pred_ours", "target": "target_ours"})
        base = data[base_name][seed].rename(columns={"pred": "pred_base", "target": "target_base"})
        merged = ours[["image_id", "target_ours", "pred_ours"]].merge(
            base[["image_id", "target_base", "pred_base"]], on="image_id", how="inner"
        )
        if merged.empty:
            raise ValueError(f"No common image_id between {ours_name} and {base_name}, seed={seed}")
        max_diff = float(np.nanmax(np.abs(merged["target_ours"].to_numpy() - merged["target_base"].to_numpy())))
        if max_diff > 1e-6:
            print(f"Warning: target mismatch for {base_name}, seed={seed}, max diff={max_diff:.6g}. Using ours target.")
        merged["target"] = merged["target_ours"]
        ids = set(merged["image_id"].astype(str))
        common_all = ids if common_all is None else common_all & ids
        aligned[seed] = merged[["image_id", "target", "pred_ours", "pred_base"]].copy()

    if not common_all:
        raise ValueError(f"No common image_id across all seeds for baseline {base_name}")
    common_all = sorted(common_all)

    for seed in seeds:
        df = aligned[seed]
        df = df[df["image_id"].isin(common_all)].copy()
        df["image_id"] = pd.Categorical(df["image_id"], categories=common_all, ordered=True)
        df = df.sort_values("image_id").reset_index(drop=True)
        df["image_id"] = df["image_id"].astype(str)
        aligned[seed] = df
    return aligned


def compute_point_rows(aligned: Dict[int, pd.DataFrame], base_name: str, metrics: List[str]) -> List[Dict]:
    rows = []
    for seed, df in aligned.items():
        y = df["target"].to_numpy(dtype=np.float64)
        po = df["pred_ours"].to_numpy(dtype=np.float64)
        pb = df["pred_base"].to_numpy(dtype=np.float64)
        for metric in metrics:
            ours_m = metric_value(y, po, metric)
            base_m = metric_value(y, pb, metric)
            rows.append({
                "baseline": base_name,
                "seed": seed,
                "metric": metric,
                "n_images": len(df),
                "ours_metric": ours_m,
                "baseline_metric": base_m,
                "delta_positive_means_ours_better": delta_value(ours_m, base_m, metric),
            })
    return rows


def paired_bootstrap(aligned: Dict[int, pd.DataFrame], base_name: str, metrics: List[str], n_boot: int, rng: np.random.Generator) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    seeds = list(aligned.keys())
    n = len(next(iter(aligned.values())))
    arrs = {}
    for seed, df in aligned.items():
        arrs[seed] = {
            "target": df["target"].to_numpy(dtype=np.float64),
            "ours": df["pred_ours"].to_numpy(dtype=np.float64),
            "base": df["pred_base"].to_numpy(dtype=np.float64),
        }

    rows = []
    delta_store = {}
    for metric in metrics:
        deltas = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            seed_deltas = []
            for seed in seeds:
                y = arrs[seed]["target"][idx]
                po = arrs[seed]["ours"][idx]
                pb = arrs[seed]["base"][idx]
                ours_m = metric_value(y, po, metric)
                base_m = metric_value(y, pb, metric)
                seed_deltas.append(delta_value(ours_m, base_m, metric))
            deltas[b] = float(np.mean(seed_deltas))

        ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
        lower_tail = (np.sum(deltas <= 0.0) + 1) / (n_boot + 1)
        upper_tail = (np.sum(deltas >= 0.0) + 1) / (n_boot + 1)
        p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))

        rows.append({
            "baseline": base_name,
            "metric": metric,
            "n_images": n,
            "n_seeds": len(seeds),
            "n_boot": n_boot,
            "delta_mean_positive_means_ours_better": float(np.mean(deltas)),
            "delta_median_positive_means_ours_better": float(np.median(deltas)),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "p_value": float(p_value),
            "significant_ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
            "direction": "baseline - ours" if metric in ("mse", "mae") else "ours - baseline",
        })
        delta_store[f"{safe_name(base_name)}__{metric}"] = deltas
    return rows, delta_store


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def main():
    args = parse_args()
    print("=" * 80)
    print("Paired bootstrap significance test")
    print(f"Ours      : {args.ours}")
    print(f"Baselines : {args.baselines}")
    print(f"Seeds     : {args.seeds}")
    print(f"Metrics   : {args.metrics}")
    print(f"n_boot    : {args.n_boot}")
    print("=" * 80)

    data = load_predictions(args)

    # ============================================================
    # 一致性自检：保证和 baseline 训练脚本使用
    #   同一个 test split / 同一个 image_id 集合 /
    #   同一个 label_raw scale / 同一个 pred_raw scale。
    # ============================================================
    print("\n[checkpoint/split/scale 自检]")
    print(f"{'model':<20s}{'seed':<6s}{'n_rows':<8s}"
          f"{'label_min':>10s}{'label_max':>10s}{'label_mean':>11s}"
          f"{'pred_min':>10s}{'pred_max':>10s}{'pred_mean':>11s}")
    reference_model = args.ours
    reference_seed = args.seeds[0]
    ref_ids = set(data[reference_model][reference_seed]["image_id"].astype(str))
    for m, seeds_by_m in data.items():
        for seed, df in seeds_by_m.items():
            y = pd.to_numeric(df["target"], errors="coerce").to_numpy()
            p = pd.to_numeric(df["pred"], errors="coerce").to_numpy()
            print(f"{m:<20s}{seed:<6d}{len(df):<8d}"
                  f"{np.nanmin(y):>10.3f}{np.nanmax(y):>10.3f}{np.nanmean(y):>11.3f}"
                  f"{np.nanmin(p):>10.3f}{np.nanmax(p):>10.3f}{np.nanmean(p):>11.3f}")
            # 检查 image_id 是否与 reference 相同（集合比较，不要求顺序）
            cur_ids = set(df["image_id"].astype(str))
            if cur_ids != ref_ids:
                only_ref = sorted(ref_ids - cur_ids)[:5]
                only_cur = sorted(cur_ids - ref_ids)[:5]
                print(f"  ⚠️  image_id 与 reference({reference_model}_seed{reference_seed}) 不一致: "
                      f"|ref-cur|={len(ref_ids - cur_ids)}, |cur-ref|={len(cur_ids - ref_ids)}; "
                      f"ref-only 示例: {only_ref}, cur-only 示例: {only_cur}")
            # 检查 label 范围是否合理（AVA 应在 1~10 之间）
            if np.nanmin(y) < 0.5 or np.nanmax(y) > 10.5:
                print(f"  ⚠️  label_raw 超出 AVA 1~10 范围: min={np.nanmin(y):.3f}, max={np.nanmax(y):.3f}")
            # 检查 pred 范围是否合理
            if np.nanmin(p) < 0.5 or np.nanmax(p) > 10.5:
                print(f"  ⚠️  pred_raw 超出 AVA 1~10 范围: min={np.nanmin(p):.3f}, max={np.nanmax(p):.3f}")
    print("=" * 80)

    rng = np.random.default_rng(args.bootstrap_seed)

    all_rows = []
    point_rows = []
    all_deltas = {}
    for base in args.baselines:
        print(f"\nComparing {args.ours} vs {base}")
        aligned = align_for_comparison(data, args.ours, base, args.seeds)
        n_images = len(next(iter(aligned.values())))
        print(f"Common test images across all seeds: {n_images}")
        point_rows.extend(compute_point_rows(aligned, base, args.metrics))
        rows, deltas = paired_bootstrap(aligned, base, args.metrics, args.n_boot, rng)
        all_rows.extend(rows)
        all_deltas.update(deltas)
        for row in rows:
            print(
                f"{row['metric'].upper():4s}: delta={row['delta_mean_positive_means_ours_better']:.6f}, "
                f"95% CI=[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}], "
                f"p={row['p_value']:.6g}, sig={row['significant_ci_excludes_zero']}"
            )

    result_df = pd.DataFrame(all_rows)
    point_df = pd.DataFrame(point_rows)
    if not result_df.empty:
        result_df["p_holm_all_tests"] = holm_bonferroni(result_df["p_value"].to_numpy())
        result_df["significant_holm_0.05"] = result_df["p_holm_all_tests"] < 0.05

    result_df.to_csv(args.out_csv, index=False)
    point_df.to_csv(args.point_csv, index=False)
    if args.save_deltas_npz:
        np.savez(args.save_deltas_npz, **all_deltas)

    print("\n" + "=" * 80)
    print(f"Saved bootstrap results: {args.out_csv}")
    print(f"Saved point metrics     : {args.point_csv}")
    if args.save_deltas_npz:
        print(f"Saved raw deltas        : {args.save_deltas_npz}")
    print("=" * 80)


if __name__ == "__main__":
    main()
    # 防止 Windows 下双击运行时 cmd 窗口一闪而过
    try:
        _ = input("\n[done] 按回车退出...")
    except EOFError:
        pass
