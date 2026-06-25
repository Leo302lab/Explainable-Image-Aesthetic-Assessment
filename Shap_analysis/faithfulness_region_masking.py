# faithfulness_region_masking.py
from __future__ import annotations

import os
import sys
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

# ---- project imports ----
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils.config as cfg  # noqa: E402
from utils.config import (  # noqa: E402
    DEVICE,
    IMAGE_DIR,
    SEMANTIC_MASK_DIR,
    FEATURE_DIR,
    MEAN,
    STD,
    IMAGE_SIZE,
    SEG_NUM_CLASSES,
    SEMANTIC_LABELS,
    ModelConfig,
)
from models.full_model import FullAestheticModel  # noqa: E402


# ============================================================================
# 1. Configuration
# ============================================================================

SEED = 42

BLUR_RADIUS = 15
MIN_REGION_PIXELS = 64
BOOTSTRAP_B = 5000
DEFAULT_MAX_IMAGES = 500

TEST_CSV = os.path.join(FEATURE_DIR, "test_features.csv")
REGION_SHAP_CSV = os.path.join(PROJECT_ROOT, "Shap_analysis", "region_shap_values.csv")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "SSG_CAF_model_region_balanced_v4.pth")

# ---- SHAP auto-generation parameters ----
SHAP_MAX_SAMPLES = 256      # compute SHAP over first N test-set samples
SHAP_BG_SAMPLES = 128        # background samples for DeepExplainer
SHAP_DATALOADER_BATCH = 8

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Shap_analysis", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "faithfulness_region_perturbation.csv")
OUTPUT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "faithfulness_region_perturbation_summary.csv")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================================
# 2. Model loading
# ============================================================================

def load_ssgcaf_model(checkpoint_path: str) -> FullAestheticModel:
    """
    Load the pretrained SSG-CAF model, using the SAME checkpoint layout as
    train_full_model.py: prefer ckpt["model_state_dict"], fall back to a bare
    state_dict, and ignore DDP "module." prefix. Returns eval mode on DEVICE.
    """
    model_cfg = ModelConfig()
    model = FullAestheticModel(model_cfg)

    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        state_dict = ckpt
        if isinstance(ckpt, dict):
            for k in ("model_state_dict", "state_dict", "model", "net"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    state_dict = ckpt[k]
                    break

        cleaned = {}
        for k, v in state_dict.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            cleaned[nk] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[model] loaded: {checkpoint_path}")
        print(f"       missing keys: {len(missing)}  |  unexpected keys: {len(unexpected)}")
    else:
        print(f"[warn] checkpoint NOT found: {checkpoint_path}")
        print("       Running with randomly initialized weights (DEBUG ONLY).")

    model.to(DEVICE)
    model.eval()
    return model


# ============================================================================
# 3. Image / mask transforms for model forward
# ============================================================================

_IMAGE_TF = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def preprocess_image(image_pil: Image.Image) -> torch.Tensor:
    """Resize + normalize + add batch dim + move to device."""
    return _IMAGE_TF(image_pil).unsqueeze(0).to(DEVICE)


def preprocess_mask(mask_np: np.ndarray,
                    target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
                    num_classes: int = SEG_NUM_CLASSES) -> torch.Tensor:
    """
    Resize mask with NEAREST-NEIGHBOR only (class IDs must NOT be interpolated).
    Class IDs are clipped to [0, num_classes - 1] so that the internal
    F.one_hot call in FullAestheticModel never raises index-out-of-range.
    Returns (1, H, W) long tensor on DEVICE.
    """
    mask_clipped = np.clip(np.asarray(mask_np, dtype=np.int64), 0, num_classes - 1)
    pil = Image.fromarray(mask_clipped.astype(np.uint8))
    pil_resized = pil.resize(target_size, Image.NEAREST)
    arr = np.array(pil_resized, dtype=np.int64)
    return torch.from_numpy(arr).long().unsqueeze(0).to(DEVICE)


# ============================================================================
# 4. Perturbation + prediction
# ============================================================================

def perturb_region_blur(
    image_pil: Image.Image,
    region_mask: np.ndarray,
    blur_radius: int = BLUR_RADIUS,
) -> Image.Image:
    """
    Blur only the pixels inside region_mask; leave everything else unchanged.
    Operates on the ORIGINAL image resolution -- the model's own resizing will
    handle it afterward.
    """
    assert region_mask.shape[:2] == (image_pil.height, image_pil.width), (
        f"mask/image shape mismatch: mask={region_mask.shape}, "
        f"img=({image_pil.height},{image_pil.width})"
    )
    image_np = np.array(image_pil).copy()
    blurred_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred_np = np.array(blurred_pil)
    image_np[region_mask] = blurred_np[region_mask]
    return Image.fromarray(image_np.astype(np.uint8))


@torch.no_grad()
def predict_score(
    model: FullAestheticModel,
    image_pil: Image.Image,
    mask_np: np.ndarray,
) -> float:
    """
    Forward-pass the SSG-CAF model with a FIXED semantic mask, and
    handcrafted features disabled -> the prediction only depends on the
    image content + the semantic segmentation mask. This isolates the
    effect of perturbation to the deep vision pathway.

    Returns predicted aesthetic score on AVA 1-10 scale.
    """
    x = preprocess_image(image_pil)
    seg_masks = preprocess_mask(mask_np)

    out = model(
        x=x,
        handcrafted_global=None,
        handcrafted_region=None,
        x_vit=None,
        seg_masks=seg_masks,
    )

    score_tensor = out.get("final_score_10")
    if score_tensor is None:
        score_tensor = out.get("final_score")
    if score_tensor is None:
        raise RuntimeError(
            "FullAestheticModel forward did not return final_score / final_score_10. "
            f"Available keys: {list(out.keys())}"
        )
    return float(score_tensor.detach().cpu().item())


# ============================================================================
# 5. Image / mask I/O helpers
# ============================================================================

def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_uint8(path: str) -> np.ndarray:
    mask = Image.open(path)
    if mask.mode != "L":
        mask = mask.convert("L")
    return np.array(mask, dtype=np.uint8)


def resolve_image_path(image_id: str) -> Optional[str]:
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = os.path.join(IMAGE_DIR, f"{image_id}{ext}")
        if os.path.isfile(p):
            return p
    return None


def resolve_mask_path(image_id: str) -> Optional[str]:
    for ext in [".png", ".jpg", ".JPG"]:
        p = os.path.join(SEMANTIC_MASK_DIR, f"{image_id}{ext}")
        if os.path.isfile(p):
            return p
    return None


# ============================================================================
# 7. Region ID helpers (SHAP -> mask correspondence)
# ============================================================================

def get_valid_region_ids(mask_np: np.ndarray, candidate_ids: List[int]) -> List[int]:
    """
    From a set of candidate region IDs (from SHAP table), keep only those
    that actually cover >= MIN_REGION_PIXELS in the current mask.
    """
    valid = []
    for rid in candidate_ids:
        if int(np.sum(mask_np == rid)) >= MIN_REGION_PIXELS:
            valid.append(rid)
    return valid


# ============================================================================
# 8. Paired-bootstrap significance test
# ============================================================================

def paired_bootstrap(
    diff_values: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_B,
    seed: int = SEED,
) -> Tuple[float, float, float, float]:
    """
    Bootstrap on paired difference: diff_values = Delta_top - Delta_rand.
    If mean_diff > 0 and p-value is small, the model is FAITHFUL to SHAP.

    Returns: (mean_diff, ci_low, ci_high, one_sided_p_value)
    """
    rng = np.random.default_rng(seed)
    diff_values = np.asarray(diff_values, dtype=np.float64)
    n = len(diff_values)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0

    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diff_values[idx].mean()

    mean_diff = float(diff_values.mean())
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    p_value = float((np.sum(boot_means <= 0) + 1) / (n_bootstrap + 1))
    return mean_diff, float(ci_low), float(ci_high), p_value


# ============================================================================
# 9. Summary helper (mean +/- std with 4-decimal digits)
# ============================================================================

def _fmt4(v: float) -> str:
    return f"{v:.4f}"


def compute_and_print_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregated stats + bootstrap test + pretty print."""
    if len(result_df) == 0:
        print("\n[skip] No valid samples.")
        return pd.DataFrame()

    diff_values = result_df["delta_diff"].values.astype(np.float64)
    mean_diff, ci_low, ci_high, p_value = paired_bootstrap(diff_values)

    summary: Dict = {
        "n_images": int(len(result_df)),
        "delta_top_mean": float(result_df["delta_top"].mean()),
        "delta_top_sd":   float(result_df["delta_top"].std()),
        "delta_top_median": float(result_df["delta_top"].median()),
        "delta_rand_mean": float(result_df["delta_rand"].mean()),
        "delta_rand_sd":  float(result_df["delta_rand"].std()),
        "delta_rand_median": float(result_df["delta_rand"].median()),
        "mean_difference": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value_one_sided": p_value,
    }

    for rname in sorted(set(result_df["top_region_name"].dropna().tolist())):
        subset = result_df[result_df["top_region_name"] == rname]
        if len(subset) < 10:
            continue
        d = subset["delta_diff"].values.astype(np.float64)
        md, _, _, pv = paired_bootstrap(d)
        summary[f"top_{rname}_n"] = int(len(subset))
        summary[f"top_{rname}_mean_diff"] = md
        summary[f"top_{rname}_p"] = pv

    summary_df = pd.DataFrame([summary])

    print("\n" + "=" * 64)
    print("  Region-Level Perturbation Test Summary")
    print("=" * 64)
    print(f"  N images          : {summary['n_images']}")
    print(f"  Delta_top  (mean +/- sd) : {_fmt4(summary['delta_top_mean'])} +/- {_fmt4(summary['delta_top_sd'])}")
    print(f"  Delta_rand (mean +/- sd) : {_fmt4(summary['delta_rand_mean'])} +/- {_fmt4(summary['delta_rand_sd'])}")
    print(f"  Mean difference   : {_fmt4(summary['mean_difference'])}")
    print(f"  95% CI            : [{_fmt4(summary['ci_low'])}, {_fmt4(summary['ci_high'])}]")
    print(f"  one-sided p-value : {summary['p_value_one_sided']:.6f}")

    rnames = sorted(set(
        k[len("top_"):-len("_mean_diff")] for k in summary.keys()
        if k.startswith("top_") and k.endswith("_mean_diff")
    ))
    for rname in rnames:
        n = summary.get(f"top_{rname}_n", 0)
        md = summary.get(f"top_{rname}_mean_diff", 0.0)
        pv = summary.get(f"top_{rname}_p", 1.0)
        print(f"    top={rname:<10s} n={n:>4}  mean_diff={_fmt4(md)}  p={pv:.6f}")

    if summary["mean_difference"] > 0 and summary["p_value_one_sided"] < 0.05:
        interpret = "Faithful: top-SHAP region perturbation shifts prediction significantly more than random."
    elif summary["mean_difference"] > 0 and summary["p_value_one_sided"] < 0.1:
        interpret = "Weakly faithful: trend exists but p-value is marginal."
    else:
        interpret = "NOT faithful: no significant difference between top-SHAP and random region perturbations."
    print(f"\n  Interpretation: {interpret}")
    print("=" * 64 + "\n")
    return summary_df


# ============================================================================
# 10. Main loop
# ============================================================================

def main(
    checkpoint_path: str = CHECKPOINT_PATH,
    test_csv_path: str = TEST_CSV,
    shap_csv_path: str = REGION_SHAP_CSV,
    max_images: int = DEFAULT_MAX_IMAGES,
    output_csv: str = OUTPUT_CSV,
    output_summary_csv: str = OUTPUT_SUMMARY_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = load_ssgcaf_model(checkpoint_path)

    # NOTE: Running in PURE VISION PATHWAY mode.
    # handcrafted_global / handcrafted_region are held at None for all calls.
    # This isolates prediction shifts to the deep feature extractor +
    # cross-attention fusion, which is the natural target of a SHAP
    # faithfulness test.

    if not os.path.isfile(shap_csv_path):
        raise FileNotFoundError(
            f"SHAP CSV not found: {shap_csv_path}\n"
            f"Expected columns: image_id, region_id, region_name, shap_value"
        )
    shap_df = pd.read_csv(shap_csv_path)
    shap_df["image_id"] = shap_df["image_id"].astype(str).str.strip()

    image_ids: List[str] = []
    if os.path.isfile(test_csv_path):
        _df = pd.read_csv(test_csv_path)
        id_col = None
        for c in ["image_id", "id", "image", "img_id"]:
            if c in _df.columns:
                id_col = c
                break
        if id_col is not None:
            image_ids = [str(x).strip() for x in _df[id_col].tolist()]
    if not image_ids:
        try:
            for fn in sorted(os.listdir(IMAGE_DIR)):
                stem = Path(fn).stem
                if resolve_mask_path(stem) is not None:
                    image_ids.append(stem)
        except FileNotFoundError:
            pass
    if max_images > 0:
        image_ids = image_ids[:max_images]

    print(f"\n[config] images to process    : {len(image_ids)}")
    print(f"[config] model checkpoint  : {checkpoint_path}")
    print(f"[config] test CSV (img list) : {test_csv_path}")
    print(f"[config] SHAP CSV            : {shap_csv_path}")
    print(f"[config] image dir             : {IMAGE_DIR}")
    print(f"[config] mask dir              : {SEMANTIC_MASK_DIR}")
    print(f"[config] mode                  : PURE VISION PATHWAY (no handcrafted features)")
    print(f"[config] perturbation blur rad : {BLUR_RADIUS}")
    print(f"[config] bootstrap iterations : {BOOTSTRAP_B}")
    print(f"[config] min region pixels   : {MIN_REGION_PIXELS}")
    print()

    results: List[Dict] = []
    skip_missing_img = 0
    skip_missing_mask = 0
    skip_no_shap = 0
    skip_few_regions = 0

    for image_id in tqdm(image_ids, desc="Faithfulness test", ncols=80):
        image_path = resolve_image_path(image_id)
        if image_path is None:
            skip_missing_img += 1
            continue
        mask_path = resolve_mask_path(image_id)
        if mask_path is None:
            skip_missing_mask += 1
            continue

        image_pil = load_image_rgb(image_path)
        mask_np = load_mask_uint8(mask_path)

        img_shap = shap_df[shap_df["image_id"] == image_id].copy()
        if len(img_shap) == 0:
            stem = Path(image_id).stem
            img_shap = shap_df[shap_df["image_id"] == stem].copy()
        if len(img_shap) == 0:
            skip_no_shap += 1
            continue

        candidate_ids = sorted(set(int(x) for x in img_shap["region_id"].values))
        valid_ids = get_valid_region_ids(mask_np, candidate_ids)
        if len(valid_ids) < 2:
            skip_few_regions += 1
            continue

        img_shap = img_shap[img_shap["region_id"].astype(int).isin(valid_ids)].copy()
        if len(img_shap) == 0:
            skip_few_regions += 1
            continue

        # top-SHAP region
        img_shap["abs_shap"] = img_shap["shap_value"].abs()
        top_row = img_shap.sort_values("abs_shap", ascending=False).iloc[0]
        top_region_id = int(top_row["region_id"])
        top_region_name = str(top_row.get("region_name",
                                            SEMANTIC_LABELS.get(top_region_id, f"r{top_region_id}")))
        top_shap_value = float(top_row["shap_value"])

        # random non-top region
        random_candidates = [rid for rid in valid_ids if rid != top_region_id]
        rand_region_id = random.choice(random_candidates)
        rand_row = img_shap[img_shap["region_id"].astype(int) == rand_region_id]
        if len(rand_row) > 0:
            rand_region_name = str(rand_row.iloc[0].get("region_name",
                                                        SEMANTIC_LABELS.get(rand_region_id, f"r{rand_region_id}")))
            rand_shap_value = float(rand_row.iloc[0]["shap_value"])
        else:
            rand_region_name = SEMANTIC_LABELS.get(rand_region_id, f"r{rand_region_id}")
            rand_shap_value = float("nan")

        # PURE VISION PATHWAY: no handcrafted features; only image content drives the score.
        original_score = predict_score(model, image_pil, mask_np)

        top_region_mask = (mask_np == top_region_id)
        top_perturbed_img = perturb_region_blur(image_pil, top_region_mask, blur_radius=BLUR_RADIUS)
        top_score = predict_score(model, top_perturbed_img, mask_np)

        rand_region_mask = (mask_np == rand_region_id)
        rand_perturbed_img = perturb_region_blur(image_pil, rand_region_mask, blur_radius=BLUR_RADIUS)
        rand_score = predict_score(model, rand_perturbed_img, mask_np)

        delta_top = float(abs(original_score - top_score))
        delta_rand = float(abs(original_score - rand_score))
        delta_diff = float(delta_top - delta_rand)

        results.append({
            "image_id": image_id,
            "original_score": original_score,
            "top_region_id": top_region_id,
            "top_region_name": top_region_name,
            "top_shap_value": top_shap_value,
            "top_perturbed_score": top_score,
            "delta_top": delta_top,
            "random_region_id": rand_region_id,
            "random_region_name": rand_region_name,
            "random_shap_value": rand_shap_value,
            "random_perturbed_score": rand_score,
            "delta_rand": delta_rand,
            "delta_diff": delta_diff,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)

    print(f"\n[done] processed images : {len(result_df)}")
    if skip_missing_img: print(f"       skipped (no img) : {skip_missing_img}")
    if skip_missing_mask: print(f"       skipped (no mask): {skip_missing_mask}")
    if skip_no_shap: print(f"       skipped (no shap): {skip_no_shap}")
    if skip_few_regions: print(f"       skipped (<2 reg): {skip_few_regions}")

    summary_df = compute_and_print_summary(result_df)
    if not summary_df.empty:
        summary_df.to_csv(output_summary_csv, index=False)
    return result_df, summary_df

# ============================================================================
# 11. Entry point (CLI)
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Region-level faithfulness test: perturb high-SHAP regions vs. random regions, with ORIGINAL semantic mask held fixed.",
    )
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="Path to SSG-CAF checkpoint.")
    parser.add_argument("--test_csv", type=str, default=TEST_CSV,
                        help="CSV providing test-set image_ids (column: image_id / id).")
    parser.add_argument("--shap_csv", type=str, default=REGION_SHAP_CSV,
                        help="CSV with per-image region-level SHAP values.")
    parser.add_argument("--max_images", type=int, default=DEFAULT_MAX_IMAGES,
                        help=f"If > 0, only process first N images (default: {DEFAULT_MAX_IMAGES}).")
    parser.add_argument("--blur_radius", type=int, default=BLUR_RADIUS,
                        help="Gaussian blur radius for region perturbation.")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B,
                        help="Number of bootstrap iterations.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Directory for output CSVs.")
    args = parser.parse_args()

    BLUR_RADIUS = args.blur_radius
    BOOTSTRAP_B = args.bootstrap
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "faithfulness_region_perturbation.csv")
    out_sum = os.path.join(args.output_dir, "faithfulness_region_perturbation_summary.csv")

    main(
        checkpoint_path=args.checkpoint,
        test_csv_path=args.test_csv,
        shap_csv_path=args.shap_csv,
        max_images=args.max_images,
        output_csv=out_csv,
        output_summary_csv=out_sum,
    )
