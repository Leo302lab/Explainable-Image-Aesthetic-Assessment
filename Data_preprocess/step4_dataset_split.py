import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 定义基础目录
BASE_DIR = r"/root/autodl-tmp"

# 定义RAW_DATA_DIR（用户指定的AVA数据集目录）
RAW_DATA_DIR = os.path.join(BASE_DIR, "AVA＿dataset")

# 定义其他路径（使用绝对路径和正确的分隔符）
CLEANED_DIR = os.path.join(BASE_DIR, "Data_preprocess/cleaned")  # 绝对路径，Linux风格
CLEANED_IMAGE_DIR = CLEANED_DIR  # 保持一致性
SPLITS_DIR = os.path.join(BASE_DIR, "Data_preprocess/splits")  # 绝对路径
ANNOT_CSV_PATH = os.path.join(RAW_DATA_DIR, "cleaned_ava.csv")  # step2生成
SPLIT_SAVE_DIR = os.path.join(RAW_DATA_DIR, "splits")  # 划分结果保存目录

# 验证关键文件是否存在
print("🔧 路径验证:")
print(f"   BASE_DIR: {BASE_DIR} {'✅' if os.path.exists(BASE_DIR) else '❌'}")
print(f"   RAW_DATA_DIR: {RAW_DATA_DIR} {'✅' if os.path.exists(RAW_DATA_DIR) else '❌'}")
print(f"   ANNOT_CSV_PATH: {ANNOT_CSV_PATH} {'✅' if os.path.exists(ANNOT_CSV_PATH) else '❌'}")

# 如果文件存在于Linux路径但在Windows上运行，提供提示
if os.name == 'nt' and not os.path.exists(BASE_DIR):
    print("⚠️ 注意：您正在Windows上运行，但配置的是Linux路径")
    print("⚠️ 请确认您是否使用了Linux环境（如WSL）或需要调整路径")

# 针对5万张数据集的合理比例
TRAIN_RATIO = 0.7    # 训练集 80%（约4.1万张）
VAL_RATIO = 0.1      # 验证集 10%（约5100张）
TEST_RATIO = 0.2     # 测试集 10%（约5100张）
RANDOM_STATE = 42  # 可复现性


def create_split_directories():
    """创建划分结果保存目录"""
    os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)
    print(f"划分结果保存目录：{SPLIT_SAVE_DIR}")


def load_and_validate_data():
    """加载清洗后的标注，验证图像文件有效性"""
    # 检查清洗后的标注文件
    if not os.path.exists(ANNOT_CSV_PATH):
        raise FileNotFoundError(f"未找到清洗标注文件：{ANNOT_CSV_PATH}，请先运行step2_data_cleaning.py")

    df = pd.read_csv(ANNOT_CSV_PATH)
    print(f"加载清洗后数据：{len(df)} 条记录")
    print(f"图像验证目录：{CLEANED_DIR}")
    
    # 统计cleaned目录中的实际图片数量
    if os.path.exists(CLEANED_DIR):
        actual_images = [f[:-4] for f in os.listdir(CLEANED_DIR) if f.endswith('.jpg')]
        print(f"cleaned目录中实际图片数量：{len(actual_images)} 张")
    else:
        print(f"❌ 警告：cleaned目录不存在：{CLEANED_DIR}")
        actual_images = []

    # 验证图像是否存在（过滤无效记录）
    valid_image_ids = []
    missing_count = 0
    
    for idx, row in tqdm(df.iterrows(), desc="验证图像有效性"):
        img_id = str(row['image_id']).zfill(5)  # 确保是5位数字格式
        img_path = os.path.join(CLEANED_DIR, f"{img_id}.jpg")
        
        if os.path.exists(img_path):
            valid_image_ids.append(row['image_id'])
        else:
            missing_count += 1
            # 每1000个缺失图片显示一次进度
            if missing_count % 1000 == 0:
                print(f"   已发现 {missing_count} 张缺失图片...")

    df_valid = df[df['image_id'].isin(valid_image_ids)].reset_index(drop=True)
    invalid_count = len(df) - len(df_valid)
    
    print(f"\n图像验证结果：")
    print(f"   原始数据：{len(df)} 条记录")
    print(f"   有效数据：{len(df_valid)} 条记录（{len(df_valid)/len(df)*100:.1f}%）")
    print(f"   无效数据：{invalid_count} 条记录（{invalid_count/len(df)*100:.1f}%）")
    print(f"   预期应匹配：{len(actual_images)} 张实际图片")

    if len(df_valid) == 0:
        raise ValueError("❌ 没有找到有效图片！请检查CLEANED_DIR路径是否正确")
    
    return df_valid


def stratified_split_by_score(df):
    """基于评分进行分层抽样划分数据集"""
    # 在函数内部直接使用全局变量
    train_ratio = TRAIN_RATIO
    val_ratio = VAL_RATIO
    test_ratio = TEST_RATIO#""按美学评分分层抽样，保证各子集评分分布一致"""
    # 检查评分范围
    print(f"评分范围：{df['mean_score'].min():.2f} ~ {df['mean_score'].max():.2f}")

    # 方法1：使用等频分箱（分位数分箱），避免边界值问题
    try:
        # 尝试使用分位数分箱，每个区间样本数大致相同
        df['score_bin'] = pd.qcut(
            df['mean_score'],
            q=5,
            labels=['q1', 'q2', 'q3', 'q4', 'q5'],
            duplicates='drop'
        )
        print("使用等频分箱（分位数分箱）")
    except ValueError:
        # 如果分位数分箱失败（如数据分布问题），使用等宽分箱
        print("分位数分箱失败，使用等宽分箱")
        # 使用更安全的边界值，避免NaN
        bins = [df['mean_score'].min() - 0.1] + list(np.linspace(df['mean_score'].min(), df['mean_score'].max(), 6))[1:]
        if len(bins) > 6:  # 确保不超过6个边界
            bins = bins[:6]

        df['score_bin'] = pd.cut(
            df['mean_score'],
            bins=bins,
            labels=['bin1', 'bin2', 'bin3', 'bin4', 'bin5'],
            include_lowest=True
        )

    # 检查是否有NaN值
    nan_count = df['score_bin'].isna().sum()
    if nan_count > 0:
        print(f"警告：有 {nan_count} 个样本的评分无法分箱")

        # 显示异常评分样本
        nan_samples = df[df['score_bin'].isna()]
        print("无法分箱的样本评分（前10个）：")
        print(nan_samples[['image_id', 'mean_score']].head(10))

        # 删除包含NaN的行
        df = df.dropna(subset=['score_bin']).copy()
        print(f"已移除 {nan_count} 个无法分箱的样本")

    # 检查分箱分布
    bin_counts = df['score_bin'].value_counts().sort_index()
    print("分箱分布：")
    for bin_name, count in bin_counts.items():
        print(f"  {bin_name}: {count} 个样本 ({count / len(df) * 100:.1f}%)")

    # 第一步：划分训练集和临时集
    train_df, temp_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        random_state=RANDOM_STATE,
        stratify=df['score_bin']  # 按评分区间分层
    )

    # 第二步：划分验证集和测试集
    # 计算临时集中验证集和测试集的比例
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=RANDOM_STATE,
        stratify=temp_df['score_bin']
    )

    # 移除分层标签
    train_df = train_df.drop('score_bin', axis=1)
    val_df = val_df.drop('score_bin', axis=1)
    test_df = test_df.drop('score_bin', axis=1)

    return train_df, val_df, test_df


def save_split_results(train_df, val_df, test_df):
    """保存划分结果：ID列表 + 子集标注CSV"""

    # 1. 保存图像ID列表（供特征提取调用）
    def save_id_list(ids, filename):
        save_path = os.path.join(SPLIT_SAVE_DIR, filename)
        # 确保所有ID都是字符串类型
        str_ids = [str(id) for id in ids]
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(str_ids))
        print(f"已保存 {filename}：{len(str_ids)} 个ID")

    save_id_list(train_df['image_id'].tolist(), "train_ids.txt")
    save_id_list(val_df['image_id'].tolist(), "val_ids.txt")
    save_id_list(test_df['image_id'].tolist(), "test_ids.txt")

    # 2. 保存子集标注CSV（含评分和标签）
    train_df.to_csv(os.path.join(SPLIT_SAVE_DIR, "train_annot.csv"), index=False)
    val_df.to_csv(os.path.join(SPLIT_SAVE_DIR, "val_annot.csv"), index=False)
    test_df.to_csv(os.path.join(SPLIT_SAVE_DIR, "test_annot.csv"), index=False)
    print("子集标注CSV保存完成")


def print_split_statistics(train_df, val_df, test_df):
    """打印划分统计信息（验证分布一致性）"""
    total = len(train_df) + len(val_df) + len(test_df)
    print("\n" + "=" * 50)
    print("数据集划分统计")
    print("=" * 50)
    print(f"训练集：{len(train_df)} 条（{len(train_df) / total * 100:.1f}%）")
    print(f"验证集：{len(val_df)} 条（{len(val_df) / total * 100:.1f}%）")
    print(f"测试集：{len(test_df)} 条（{len(test_df) / total * 100:.1f}%）")

    print("\n各子集评分分布（均值±标准差）")
    print(f"训练集：{train_df['mean_score'].mean():.2f} ± {train_df['mean_score'].std():.2f}")
    print(f"验证集：{val_df['mean_score'].mean():.2f} ± {val_df['mean_score'].std():.2f}")
    print(f"测试集：{test_df['mean_score'].mean():.2f} ± {test_df['mean_score'].std():.2f}")

    # 打印评分区间分布
    print("\n各子集评分区间分布")
    # 使用统一的边界值
    score_min = min(train_df['mean_score'].min(), val_df['mean_score'].min(), test_df['mean_score'].min())
    score_max = max(train_df['mean_score'].max(), val_df['mean_score'].max(), test_df['mean_score'].max())

    # 创建5个等宽区间
    bins = np.linspace(score_min, score_max, 6)
    score_bins = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(5)]

    train_bins = pd.cut(train_df['mean_score'], bins=bins, labels=score_bins).value_counts().sort_index()
    val_bins = pd.cut(val_df['mean_score'], bins=bins, labels=score_bins).value_counts().sort_index()
    test_bins = pd.cut(test_df['mean_score'], bins=bins, labels=score_bins).value_counts().sort_index()

    df_bins = pd.DataFrame({
        "训练集": train_bins,
        "验证集": val_bins,
        "测试集": test_bins
    }).fillna(0).astype(int)
    print(df_bins)
    print("=" * 50)


def main():
    """主流程：创建目录→加载数据→分层划分→保存结果→打印统计"""
    print("开始执行数据集划分（step5）...")

    # 1. 创建保存目录
    create_split_directories()

    # 2. 加载并验证数据
    df_valid = load_and_validate_data()

    # 3. 按评分分层抽样
    train_df, val_df, test_df = stratified_split_by_score(df_valid)

    # 4. 保存划分结果
    save_split_results(train_df, val_df, test_df)

    # 5. 打印统计信息
    print_split_statistics(train_df, val_df, test_df)

    print("\n数据集划分完成！所有结果保存至：", SPLIT_SAVE_DIR)
    print("后续使用说明：")
    print("- 特征提取：step4_feature_extraction.py 会自动读取该目录下的ID列表")
    print("- 模型训练：直接加载该目录下的标注CSV和特征文件")


if __name__ == "__main__":
    main()
