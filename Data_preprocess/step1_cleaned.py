import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ================= 配置区域 =================
RAW_DATA_DIR = r"E:\aesthetic_evaluation\AVA_dataset"
AVA_TXT_PATH = os.path.join(RAW_DATA_DIR, "AVA.txt")
AVA_IMG_DIR = os.path.join(RAW_DATA_DIR, "ava_images")

CLEANED_DIR = r"E:\aesthetic_evaluation\Data_preprocess\cleaned_v3"
SPLIT_DIR = os.path.join(RAW_DATA_DIR, "split_files_v3")
OUT_CSV_PATH = os.path.join(os.path.dirname(__file__), "cleaned_ava_v3.csv")

os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

# 标签过滤（手机摄影相关标签）
FILTER_TAGS = [1, 2, 3, 4, 5, 6, 7, 8]

# ✅ 高分豁免：高分图永远保留（建议 7.0~8.0 之间取，先用 7.5）
HIGH_SCORE_KEEP = 7.5

# ✅ 是否允许“高分绕过标签过滤”
KEEP_HIGH_SCORE_EVEN_IF_NOT_TARGET_TAGS = True

# ✅ 清晰度阈值（建议配合 resize 后再调）
# 通常 resize(short_side=256) 后，阈值 5~20 都有人用；你之前10可能偏严
LAPLACIAN_THRESH = 10

# Laplacian 计算前统一 resize 的短边
LAPLACIAN_SHORT_SIDE = 256

# 可选：最低投票数过滤（避免极端不可靠样本）
MIN_TOTAL_RATINGS = 0  # 例如 20；不想过滤就保持 0

# 目标样本数（不限制就 None）
TARGET_SAMPLE_SIZE = None
RANDOM_SEED = 42

# 分桶边界（用于分层切分）
SCORE_BINS = [0, 4, 5, 6, 7, 7.5, 8, 9, 10]


def read_ava_annotations(txt_path: str) -> pd.DataFrame:
    """
    读取并解析 AVA.txt
    格式: [Index] [ImageID] [Score1]...[Score10] [Tag1]...
    """
    data = []
    print(f"正在读取标注文件: {txt_path}")

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)

        for line_num, line in tqdm(enumerate(f, 1), desc="解析AVA标注", total=total_lines):
            parts = line.strip().split()
            if len(parts) < 13:
                continue

            try:
                image_id = parts[1]
                score_counts = list(map(int, parts[2:12]))
                tags = list(map(int, parts[12:]))

                total_ratings = int(sum(score_counts))
                if total_ratings <= 0:
                    mean_score = 0.0
                    std_score = 0.0
                else:
                    weighted_sum = sum((i + 1) * c for i, c in enumerate(score_counts))
                    mean_score = float(weighted_sum / total_ratings)

                    var_sum = sum(c * ((i + 1) - mean_score) ** 2 for i, c in enumerate(score_counts))
                    std_score = float(np.sqrt(var_sum / total_ratings))

                data.append(
                    {
                        "image_id": str(image_id),
                        "mean_score": float(mean_score),
                        "std_score": float(std_score),
                        "total_ratings": total_ratings,
                        "tags": tags,
                    }
                )
            except Exception:
                # 安静跳过，避免刷屏
                continue

    df = pd.DataFrame(data)
    return df


def calculate_laplacian_variance(image_path: str, short_side: int = 256) -> float:
    """计算图像清晰度（先 resize 保证阈值稳定）"""
    try:
        img_np = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return 0.0

    if img is None:
        return 0.0

    h, w = img.shape[:2]
    # 只缩小不放大，避免小图被放大后虚高
    scale = float(short_side) / float(min(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


def _has_any_target_tags(tags_list) -> bool:
    return any(t in FILTER_TAGS for t in tags_list)


def _safe_copy_jpg(src_path: str, dst_path: str) -> bool:
    """用 imdecode + imencode 复制，兼容中文路径"""
    try:
        img_np = np.fromfile(src_path, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            return False
        cv2.imencode(".jpg", img)[1].tofile(dst_path)
        return True
    except Exception:
        return False


def stratified_split(df: pd.DataFrame):
    """按 mean_score 分桶做分层切分，尽量保证 train/val/test 都有高分"""
    df = df.copy()
    df["score_bin"] = pd.cut(df["mean_score"], bins=SCORE_BINS, include_lowest=True)

    # 如果某些桶样本太少导致 stratify 报错，就退化成更粗的分桶
    def _try_split(dfx, bins):
        dfx = dfx.copy()
        dfx["score_bin"] = pd.cut(dfx["mean_score"], bins=bins, include_lowest=True)
        vc = dfx["score_bin"].value_counts(dropna=False)
        if (vc < 2).any():
            return None  # 可能会导致 stratify 失败
        train_df, temp_df = train_test_split(
            dfx, test_size=0.3, random_state=RANDOM_SEED, stratify=dfx["score_bin"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.6667, random_state=RANDOM_SEED, stratify=temp_df["score_bin"]
        )
        return train_df, val_df, test_df

    res = _try_split(df, SCORE_BINS)
    if res is None:
        # 更粗的桶
        coarse_bins = [0, 5, 6, 7, 8, 10]
        res = _try_split(df, coarse_bins)

    if res is None:
        # 实在不行：随机切（但会打印警告）
        print("⚠️ 分层切分失败（某些分桶样本过少），退化为随机切分。建议放宽清洗或增加高分样本。")
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
        val_df, test_df = train_test_split(temp_df, test_size=0.6667, random_state=RANDOM_SEED)

    # 去掉辅助列
    train_df = train_df.drop(columns=[c for c in ["score_bin"] if c in train_df.columns])
    val_df = val_df.drop(columns=[c for c in ["score_bin"] if c in val_df.columns])
    test_df = test_df.drop(columns=[c for c in ["score_bin"] if c in test_df.columns])
    return train_df, val_df, test_df


def clean_ava_data():
    # 1) 读取标注
    df = read_ava_annotations(AVA_TXT_PATH)
    print(f"原始数据量：{len(df)}")
    if df.empty:
        print("❌ 没读到数据，请检查 AVA.txt 路径/格式。")
        return

    print(
        f"分数统计: Min={df['mean_score'].min():.2f}, "
        f"Max={df['mean_score'].max():.2f}, Mean={df['mean_score'].mean():.2f}"
    )
    print(f"高分(>= {HIGH_SCORE_KEEP})数量: {(df['mean_score'] >= HIGH_SCORE_KEEP).sum()}")

    # 2) 可选：最低投票数过滤
    if MIN_TOTAL_RATINGS > 0:
        before = len(df)
        df = df[df["total_ratings"] >= MIN_TOTAL_RATINGS].reset_index(drop=True)
        print(f"投票数过滤: {before} -> {len(df)} (min_ratings={MIN_TOTAL_RATINGS})")

    # 3) 标签过滤（但可允许高分绕过）
    print("正在过滤标签...")
    if KEEP_HIGH_SCORE_EVEN_IF_NOT_TARGET_TAGS:
        df["has_target_tags"] = df["tags"].apply(_has_any_target_tags)
        before = len(df)
        df = df[(df["has_target_tags"]) | (df["mean_score"] >= HIGH_SCORE_KEEP)].reset_index(drop=True)
        print(f"标签过滤(高分绕过): {before} -> {len(df)}")
    else:
        df["has_target_tags"] = df["tags"].apply(_has_any_target_tags)
        before = len(df)
        df = df[df["has_target_tags"]].reset_index(drop=True)
        print(f"标签过滤: {before} -> {len(df)}")

    print(f"标签过滤后高分(>= {HIGH_SCORE_KEEP})数量: {(df['mean_score'] >= HIGH_SCORE_KEEP).sum()}")

    # 4) 清晰度/完整性过滤（✅ 高分豁免）
    valid_rows = []
    print("正在检查图像清晰度与完整性（高分豁免）...")

    for _, row in tqdm(df.iterrows(), desc="过滤图像", total=len(df)):
        img_id = str(row["image_id"])
        image_path = os.path.join(AVA_IMG_DIR, f"{img_id}.jpg")
        if not os.path.exists(image_path):
            continue

        mean_score = float(row["mean_score"])

        # ✅ 高分直接保留（不走 Laplacian 淘汰）
        if mean_score >= HIGH_SCORE_KEEP:
            ok = True
        else:
            lap_var = calculate_laplacian_variance(image_path, short_side=LAPLACIAN_SHORT_SIDE)
            ok = (lap_var >= LAPLACIAN_THRESH)

        if not ok:
            continue

        # 复制到清洗目录
        save_path = os.path.join(CLEANED_DIR, f"{img_id}.jpg")
        if _safe_copy_jpg(image_path, save_path):
            valid_rows.append(row)

    df_clean = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(f"清晰度过滤后数据量：{len(df_clean)}")
    print(f"清洗后高分(>= {HIGH_SCORE_KEEP})数量: {(df_clean['mean_score'] >= HIGH_SCORE_KEEP).sum()}")

    if df_clean.empty:
        print("❌ 清洗后没有剩余数据。请放宽 LAPLACIAN_THRESH / 标签过滤 或检查路径。")
        return

    # 5) 可选：随机采样
    if TARGET_SAMPLE_SIZE is not None and len(df_clean) > TARGET_SAMPLE_SIZE:
        df_clean = df_clean.sample(n=TARGET_SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"随机采样后数据量：{len(df_clean)}")

    # 6) 分层切分（7:1:2）
    train_df, val_df, test_df = stratified_split(df_clean)
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy();     val_df["split"] = "val"
    test_df = test_df.copy();   test_df["split"] = "test"

    df_out = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 7) 保存 split id 文件
    def save_ids(dfx, filename):
        ids = dfx["image_id"].astype(str).tolist()
        with open(os.path.join(SPLIT_DIR, filename), "w", encoding="utf-8") as f:
            f.write("\n".join(ids))

    save_ids(train_df, "New.train.txt")
    save_ids(val_df, "New.val.txt")
    save_ids(test_df, "New.test.txt")

    # 8) 保存 CSV
    df_out.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 清洗完成！CSV 保存至: {OUT_CSV_PATH}")
    print(f"✅ 图片保存目录: {CLEANED_DIR}")
    print(f"✅ 切分文件目录: {SPLIT_DIR}")

    # 9) 打印各 split 的高分占比，确保 train 里真的有尾部
    def _report(name, dfx):
        total = len(dfx)
        hi = int((dfx["mean_score"] >= HIGH_SCORE_KEEP).sum())
        print(f"[{name:<5}] n={total:<6} high(>= {HIGH_SCORE_KEEP})={hi:<5} ({(hi/max(total,1))*100:.2f}%) "
              f"max={dfx['mean_score'].max():.2f} mean={dfx['mean_score'].mean():.2f}")

    print("\n=== Split 统计 ===")
    _report("train", train_df)
    _report("val", val_df)
    _report("test", test_df)

    print("\n=== Overall 分数统计 ===")
    print(df_out["mean_score"].describe())


if __name__ == "__main__":
    clean_ava_data()
