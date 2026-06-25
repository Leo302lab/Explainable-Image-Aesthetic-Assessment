# E:\aesthetic_evaluation\pythonProject2\run_all_baselines.py
"""
联合训练入口脚本
顺序执行以下 baseline 训练脚本并汇总结果：
  1. training/train_baseline.py          (ResNet50 + ViT)
  2. baseline/TAVAR/TAVAR_training.py     (TAVAR)
  3. baseline/EAT/EAT_baseline_training.py  (EAT/DAT)
  4. baseline/AVA-MLSP/MLSP_training.py   (MLSP)

用法:
  python run_all_baselines.py --seeds 42 123 2024         # 默认，所有模型所有种子
  python run_all_baselines.py --models tavar eat --seed 42  # 只跑 TAVAR 和 EAT，seed=42
  python run_all_baselines.py --models baseline tavar --all_seeds  # 只跑 baseline + TAVAR
"""

import os
import sys
import argparse
import time
import json
import importlib
import csv
import numpy as np
from datetime import datetime

# 将项目根加入 path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "baseline", "TAVAR"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "baseline", "EAT"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "baseline", "AVA-MLSP"))

# ============ 支持的模型与种子 ============
MODELS = {
    "baseline": {
        "name": "ResNet50 + ViT Baseline",
        "script": "training.train_baseline",
    },
    "tavar": {
        "name": "TAVAR",
        "script": "baseline.TAVAR.TAVAR_training",
    },
    "eat": {
        "name": "EAT (DAT)",
        "script": "baseline.EAT.EAT_baseline_training",
    },
    "mlsp": {
        "name": "AVA-MLSP",
        "script": "baseline.AVA-MLSP.MLSP_training",
    },
}

DEFAULT_SEEDS = [42, 123, 2026]

# 固定跳过任务：
# 只跳过 [1/12] ResNet50 + ViT Baseline seed=42
# 其他模型和其他 seed 均正常运行。
SKIP_TASKS = {
    ("baseline", 42),
}


def _load_module(module_path: str):
    """
    统一使用 importlib 动态加载模块：
      - 支持路径含连字符 '-' 等特殊字符（例如 baseline.AVA-MLSP.MLSP_training）
      - 运行前从 sys.modules 清理同名模块，避免全局状态污染
    """
    if module_path in sys.modules:
        del sys.modules[module_path]
    return importlib.import_module(module_path)


def _safe_main(module_name, seed, model_key, timeout=None):
    """安全调用各脚本的训练入口，统一捕获异常与格式化输出。"""
    t0 = time.time()

    # 每个模型重新加载（避免全局状态污染）
    # 全部使用 importlib.import_module()，因为 AVA-MLSP 的连字符不是合法 Python 标识符
    if model_key == "baseline":
        # train_baseline.py 的每个 seed 跑两个模型：resnet50 + vit
        # （与 train_baseline.py 的 run_all_seeds 逻辑一致）
        _mod = _load_module("training.train_baseline")

        # 从已加载模块拿默认值（与 train_baseline.py parse_args 一致）
        _epochs = getattr(_mod, "EPOCHS", 30)
        _batch_size = getattr(_mod, "BATCH_SIZE", 8)
        _accum_steps = getattr(_mod, "ACCUMULATION_STEPS", 4)
        _lr_head = getattr(_mod, "LR_HEAD", 8e-5)
        _lr_backbone = getattr(_mod, "LR_BACKBONE", 8e-6)
        _weight_decay = getattr(_mod, "WEIGHT_DECAY", 5.0e-2)
        _grad_clip = getattr(_mod, "GRADIENT_CLIP", 1.0)
        _patience = getattr(_mod, "EARLY_STOP_PATIENCE", 2)
        _img_size = getattr(_mod, "DEFAULT_IMG_SIZE", 448)

        class _Args:
            pass

        def _build_args(model_name: str) -> "_Args":
            a = _Args()
            a.seed = seed
            a.model = model_name
            a.save_name = f"baseline_{model_name}_seed{seed}.pth"
            a.img_size = _img_size
            a.batch_size = _batch_size
            a.accumulation_steps = _accum_steps
            a.lr_backbone = _lr_backbone
            a.lr_head = _lr_head
            a.weight_decay = _weight_decay
            a.epochs = _epochs
            a.gradient_clip = _grad_clip
            a.patience = _patience
            a.verify_images = False
            a.deterministic = False
            a.test_only = False
            a.all_models = False
            a.all_seeds = False
            return a

        # --- 1) ResNet50 ---
        result1 = _mod.main(args=_build_args("resnet50"))

        # --- 2) ViT-Base（重新加载模块以避免全局状态污染） ---
        _mod = _load_module("training.train_baseline")
        result2 = _mod.main(args=_build_args("vit"))

        # 合并两个模型的结果
        def _to_flat_dict(x, prefix=""):
            if isinstance(x, dict):
                return {f"{prefix}{k}" if prefix else k: v for k, v in x.items()}
            if isinstance(x, (tuple, list)) and len(x) >= 4:
                return {f"{prefix}srcc": float(x[0]), f"{prefix}plcc": float(x[1]),
                        f"{prefix}mae": float(x[2]), f"{prefix}mse": float(x[3])}
            return {f"{prefix}raw": str(x)}

        merged = {}
        merged.update(_to_flat_dict(result1, prefix="resnet50_"))
        merged.update(_to_flat_dict(result2, prefix="vit_"))
        result = merged
    elif model_key == "tavar":
        _mod = _load_module("baseline.TAVAR.TAVAR_training")
        result = _mod.main(seed=seed)
    elif model_key == "eat":
        _mod = _load_module("baseline.EAT.EAT_baseline_training")
        result = _mod.run_one_seed(seed)
    elif model_key == "mlsp":
        _mod = _load_module("baseline.AVA-MLSP.MLSP_training")
        result = _mod.run_one_seed(seed)
    else:
        raise ValueError(f"未知模型: {model_key}")

    dt = time.time() - t0

    # 统一格式化输出为 dict
    if isinstance(result, (tuple, list)):
        # TAVAR: 返回 (srcc, plcc, mae, mse, loss)
        out = {"test_srcc": float(result[0]), "test_plcc": float(result[1]),
               "test_mae": float(result[2]), "test_mse": float(result[3])}
    elif isinstance(result, dict):
        # EAT/MLSP/baseline: 返回 dict
        out = {}
        for k, v in result.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
    else:
        out = {"raw": str(result)}

    out["runtime_seconds"] = round(dt, 1)
    out["seed"] = seed
    out["model"] = model_key
    return out


def _safe_float(v):
    """Try converting value to float; return None if unavailable."""
    if v is None:
        return None
    try:
        x = float(v)
        if x != x:  # NaN
            return None
        return x
    except (TypeError, ValueError):
        return None


def _first_metric(result: dict, keys: list[str]):
    for k in keys:
        if k in result:
            v = _safe_float(result.get(k))
            if v is not None:
                return v
    return None


def _extract_prefixed_metric(result: dict, prefix: str, metric: str):
    """Extract metrics from combined baseline result, e.g. resnet50_test_srcc."""
    key_map = {
        "srcc": [f"{prefix}test_srcc", f"{prefix}spearman", f"{prefix}srcc", f"{prefix}val_srcc"],
        "plcc": [f"{prefix}test_plcc", f"{prefix}pearson", f"{prefix}plcc", f"{prefix}val_pcc"],
        "mae": [f"{prefix}test_mae", f"{prefix}mae"],
        "mse": [f"{prefix}test_mse", f"{prefix}mse"],
    }
    return _first_metric(result, key_map[metric])


def _extract_general_metric(result: dict, metric: str):
    key_map = {
        "srcc": ["test_srcc", "spearman", "srcc", "val_srcc"],
        "plcc": ["test_plcc", "pearson", "plcc", "val_pcc"],
        "mae": ["test_mae", "mae"],
        "mse": ["test_mse", "mse"],
    }
    return _first_metric(result, key_map[metric])


def expand_results_for_summary(results):
    """Expand raw results into per-model rows.

    Special case:
    - model='baseline' contains two sub-models, ResNet50 and ViT-Base.
      They are split into separate rows for mean ± SD reporting.
    """
    rows = []

    for r in results:
        model_key = r.get("model", "")
        seed = r.get("seed", "?")
        note = r.get("note", "")
        runtime = _safe_float(r.get("runtime_seconds")) or 0.0

        if model_key == "baseline" and (
            any(k.startswith("resnet50_") for k in r.keys()) or any(k.startswith("vit_") for k in r.keys())
        ):
            for prefix, model_name, sub_key in [
                ("resnet50_", "ResNet50", "baseline_resnet50"),
                ("vit_", "ViT-Base", "baseline_vit"),
            ]:
                rows.append({
                    "model_key": sub_key,
                    "model_name": model_name,
                    "seed": seed,
                    "srcc": _extract_prefixed_metric(r, prefix, "srcc"),
                    "plcc": _extract_prefixed_metric(r, prefix, "plcc"),
                    "mae": _extract_prefixed_metric(r, prefix, "mae"),
                    "mse": _extract_prefixed_metric(r, prefix, "mse"),
                    "runtime_seconds": runtime / 2.0,
                    "note": note,
                    "error": r.get("error", ""),
                })
            continue

        model_name = MODELS.get(model_key, {}).get("name", model_key)

        rows.append({
            "model_key": model_key,
            "model_name": model_name,
            "seed": seed,
            "srcc": _extract_general_metric(r, "srcc"),
            "plcc": _extract_general_metric(r, "plcc"),
            "mae": _extract_general_metric(r, "mae"),
            "mse": _extract_general_metric(r, "mse"),
            "runtime_seconds": runtime,
            "note": note,
            "error": r.get("error", ""),
        })

    return rows


def _fmt_metric(v):
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def print_summary_table(results):
    """Print per-seed summary table and return expanded rows."""
    rows = expand_results_for_summary(results)

    if not rows:
        print("\n⚠️  没有可汇总的结果。")
        return []

    print("\n" + "=" * 100)
    print("📊 每个模型 × 每个随机种子的结果  (Per-seed Baseline Results)")
    print("=" * 100)
    print(f"{'模型':<18s} {'种子':>6s} {'SRCC':>8s} {'PLCC':>8s} {'MAE':>8s} {'MSE':>8s} {'耗时':>10s}  {'备注':>14s}")
    print("-" * 100)

    total_time = 0.0
    for row in rows:
        total_time += row.get("runtime_seconds", 0.0) or 0.0
        model = str(row.get("model_name", ""))
        seed = str(row.get("seed", "?"))
        runtime = float(row.get("runtime_seconds", 0.0) or 0.0)
        note = str(row.get("note", ""))

        print(
            f"{model:<18s} {seed:>6s} "
            f"{_fmt_metric(row.get('srcc')):>8s} "
            f"{_fmt_metric(row.get('plcc')):>8s} "
            f"{_fmt_metric(row.get('mae')):>8s} "
            f"{_fmt_metric(row.get('mse')):>8s} "
            f"{f'{runtime/60:.1f} min':>10s}  "
            f"{note:>14s}"
        )

    print("-" * 100)
    print(f"{'总计':<18s} {'':>6s} {'':>8s} {'':>8s} {'':>8s} {'':>8s} {f'{total_time/60:.1f} min':>10s}")
    print("=" * 100)

    return rows


def _mean_sd(values):
    values = [float(v) for v in values if v is not None]
    if len(values) == 0:
        return None, None, 0
    mean = float(sum(values) / len(values))
    if len(values) >= 2:
        sd = float(np.std(values, ddof=1))
    else:
        sd = 0.0
    return mean, sd, len(values)


def _fmt_mean_sd(mean, sd, n):
    if n == 0 or mean is None:
        return "N/A"
    return f"{mean:.4f} ± {sd:.4f}"


def print_mean_sd_summary(rows):
    """Print model-level mean ± SD across available seeds.

    Failed/skipped tasks are excluded from metric aggregation.
    The table still reports n and seeds used, so missing seeds are visible.
    """
    if not rows:
        return []

    grouped = {}
    for row in rows:
        model_key = row["model_key"]
        grouped.setdefault(model_key, {
            "model_name": row["model_name"],
            "rows": [],
        })
        grouped[model_key]["rows"].append(row)

    summary_rows = []

    print("\n" + "=" * 120)
    print("📈 三个随机种子汇总  (Mean ± SD across available successful seeds)")
    print("=" * 120)
    print(
        f"{'模型':<18s} {'n':>3s} {'Seeds used':<18s} "
        f"{'SRCC mean±sd':>18s} {'PLCC mean±sd':>18s} "
        f"{'MAE mean±sd':>18s} {'MSE mean±sd':>18s} {'失败/跳过':>10s}"
    )
    print("-" * 120)

    for model_key, pack in grouped.items():
        rows_m = pack["rows"]

        usable = [
            r for r in rows_m
            if r.get("srcc") is not None
            or r.get("plcc") is not None
            or r.get("mae") is not None
            or r.get("mse") is not None
        ]

        seeds_used = [str(r.get("seed", "?")) for r in usable]
        failed_or_skipped = len(rows_m) - len(usable)

        srcc_mean, srcc_sd, n_srcc = _mean_sd([r.get("srcc") for r in usable])
        plcc_mean, plcc_sd, n_plcc = _mean_sd([r.get("plcc") for r in usable])
        mae_mean, mae_sd, n_mae = _mean_sd([r.get("mae") for r in usable])
        mse_mean, mse_sd, n_mse = _mean_sd([r.get("mse") for r in usable])

        n = max(n_srcc, n_plcc, n_mae, n_mse)
        model_name = pack["model_name"]
        seeds_text = ",".join(seeds_used) if seeds_used else "-"

        print(
            f"{model_name:<18s} {n:>3d} {seeds_text:<18s} "
            f"{_fmt_mean_sd(srcc_mean, srcc_sd, n_srcc):>18s} "
            f"{_fmt_mean_sd(plcc_mean, plcc_sd, n_plcc):>18s} "
            f"{_fmt_mean_sd(mae_mean, mae_sd, n_mae):>18s} "
            f"{_fmt_mean_sd(mse_mean, mse_sd, n_mse):>18s} "
            f"{failed_or_skipped:>10d}"
        )

        summary_rows.append({
            "model_key": model_key,
            "model_name": model_name,
            "n": n,
            "seeds_used": seeds_used,
            "failed_or_skipped": failed_or_skipped,
            "srcc_mean": srcc_mean,
            "srcc_sd": srcc_sd,
            "plcc_mean": plcc_mean,
            "plcc_sd": plcc_sd,
            "mae_mean": mae_mean,
            "mae_sd": mae_sd,
            "mse_mean": mse_mean,
            "mse_sd": mse_sd,
        })

    print("-" * 120)
    print("说明：失败或跳过的任务不会参与 mean ± SD；n 表示实际参与统计的成功 seed 数。")
    print("=" * 120)

    return summary_rows


def save_results_json(results, out_dir, expanded_rows=None, mean_sd_rows=None):
    """保存结果 JSON。"""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"run_all_baselines_{ts}.json")

    payload = {
        "raw_results": results,
        "per_seed_rows": expanded_rows or [],
        "mean_sd_summary": mean_sd_rows or [],
        "created_at": ts,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果 JSON 已保存: {out_path}")


def save_summary_csv(expanded_rows, mean_sd_rows, out_dir):
    """保存 per-seed 和 mean±sd CSV。"""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    per_seed_path = os.path.join(out_dir, f"run_all_baselines_per_seed_{ts}.csv")
    mean_sd_path = os.path.join(out_dir, f"run_all_baselines_mean_sd_{ts}.csv")

    if expanded_rows:
        fieldnames = [
            "model_key", "model_name", "seed", "srcc", "plcc", "mae", "mse",
            "runtime_seconds", "note", "error",
        ]
        with open(per_seed_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in expanded_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"💾 Per-seed CSV 已保存: {per_seed_path}")

    if mean_sd_rows:
        fieldnames = [
            "model_key", "model_name", "n", "seeds_used", "failed_or_skipped",
            "srcc_mean", "srcc_sd", "plcc_mean", "plcc_sd",
            "mae_mean", "mae_sd", "mse_mean", "mse_sd",
        ]
        with open(mean_sd_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in mean_sd_rows:
                row2 = dict(row)
                row2["seeds_used"] = ",".join(row.get("seeds_used", []))
                writer.writerow({k: row2.get(k, "") for k in fieldnames})
        print(f"💾 Mean±SD CSV 已保存: {mean_sd_path}")


def main():
    parser = argparse.ArgumentParser(
        description="联合训练入口：顺序运行 baseline / TAVAR / EAT / MLSP 的训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python run_all_baselines.py\n"
               "  python run_all_baselines.py --models tavar eat --seed 42\n"
               "  python run_all_baselines.py --models baseline tavar --all_seeds\n"
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()), metavar="MODEL",
                        help=f"要运行的模型列表（默认全部），支持: {', '.join(MODELS.keys())}")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, metavar="SEED",
                        help=f"随机种子列表（默认: {' '.join(map(str, DEFAULT_SEEDS))}）")
    parser.add_argument("--seed", type=int, default=None, metavar="SEED",
                        help="单个种子（覆盖 --seeds）")
    parser.add_argument("--all_seeds", action="store_true",
                        help="运行所有默认种子（与 --seeds 相同效果）")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "checkpoints", "summary"),
                        help="结果 JSON 输出目录")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印将要运行的任务，不实际执行")

    args = parser.parse_args()

    # 处理 seeds
    if args.seed is not None:
        seeds = [args.seed]
    elif args.all_seeds:
        seeds = DEFAULT_SEEDS
    else:
        seeds = args.seeds

    # 去重并保持顺序
    seen = set()
    seeds = [s for s in seeds if not (s in seen or seen.add(s))]

    # 打印将要运行的任务
    print("\n" + "=" * 90)
    print("🚀 联合训练入口 (Unified Baseline Training Entry)")
    print("=" * 90)
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  将要运行的模型: {', '.join(args.models)}")
    print(f"  随机种子: {seeds}")
    print(f"  输出目录: {args.output_dir}")
    print("=" * 90 + "\n")

    all_results = []
    total_tasks = len(args.models) * len(seeds)
    task_idx = 0

    for model_key in args.models:
        model_info = MODELS[model_key]
        for seed in seeds:
            task_idx += 1
            header = f"[{task_idx}/{total_tasks}] {model_info['name']}  seed={seed}"

            # 固定跳过指定任务：baseline + seed=42
            if (model_key, seed) in SKIP_TASKS:
                print(f"⏭️  [SKIP] {header}  | reason: already completed / manually skipped")
                all_results.append({
                    "model": model_key,
                    "seed": seed,
                    "note": "⏭ skipped",
                    "runtime_seconds": 0.0,
                })
                continue

            if args.dry_run:
                print(f"🧪 (DRY RUN) {header}")
                continue

            print("\n" + "=" * 90)
            print(f"▶ {header}")
            print("=" * 90 + "\n")

            try:
                result = _safe_main(model_info["script"], seed, model_key)
                result["note"] = "✓ OK"
                all_results.append(result)
                print(f"\n✅ [{task_idx}/{total_tasks}] 完成: {model_info['name']}  seed={seed}  "
                      f"耗时 {result.get('runtime_seconds', 0)/60:.1f} min")
            except Exception as exc:
                print(f"\n❌ [{task_idx}/{total_tasks}] 失败: {model_info['name']}  seed={seed}")
                print(f"   错误: {type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "model": model_key,
                    "seed": seed,
                    "note": f"✗ {type(exc).__name__}",
                    "error": str(exc),
                    "runtime_seconds": 0.0,
                })

    # 汇总打印：per-seed + mean±sd
    expanded_rows = print_summary_table(all_results)
    mean_sd_rows = print_mean_sd_summary(expanded_rows)

    # 保存 JSON + CSV
    save_results_json(all_results, args.output_dir, expanded_rows=expanded_rows, mean_sd_rows=mean_sd_rows)
    save_summary_csv(expanded_rows, mean_sd_rows, args.output_dir)

    print("\n🎉 全部任务完成！")


if __name__ == "__main__":
    main()
