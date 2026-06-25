
import os
import torch

# ============================================================
# 基础
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SEEDS = [42, 123, 2026]  # 多个随机种子用于多次实验
PRINT_MODEL_INFO = True

# ============================================================
# 路径
# ============================================================
SEGFORMER_BASE_DIR = "/root/autodl-tmp/pythonProject2/training/models/nvidia/segformer-b0-finetuned-ade-512-512"
SEGFORMER_FINETUNE_CKPT = "/root/autodl-tmp/checkpoints/segmentation/best_segformer_finetune_V618.pth"
RESNET50_CKPT = "/root/autodl-tmp/pythonProject2/pretrained_models/resnet50.pth"
VIT_BASE_CKPT = "/root/autodl-tmp/pythonProject2/pretrained_models/vit_base.pth"

CHECKPOINT_DIR = "/root/autodl-tmp/pythonProject2/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# 数据目录
# ============================================================
FEATURE_DIR = "/root/autodl-tmp/Data_preprocess/features_Fixed_final_design_roi_hardmask"
IMAGE_DIR = "/root/autodl-tmp/Data_preprocess/cleaned"
SEMANTIC_MASK_DIR = "/root/autodl-tmp/Data_preprocess/semantic_masks"

for p in [FEATURE_DIR, IMAGE_DIR, SEMANTIC_MASK_DIR]:
    os.makedirs(p, exist_ok=True)

# ============================================================
# 数据 split
# ============================================================
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"

# ============================================================
# 分割 / 分支控制
# ============================================================
USE_OFFLINE_SEG_MASK = True
ENABLE_ONLINE_SEG_BRANCH = False
SEG_FREEZE = True
FREEZE_VIT = True
FREEZE_BATCHNORM_STATS = True
USE_ONLINE_HANDCRAFTED_EXTRACTOR = False

# ============================================================
# 图像配置
# ============================================================
IMAGE_SIZE = 224
SEG_IMAGE_SIZE = 512

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD = [0.5, 0.5, 0.5]

SEG_MEAN = [0.485, 0.456, 0.406]
SEG_STD = [0.229, 0.224, 0.225]

# ============================================================
# 标签 / 分数配置
# ============================================================
ID_COLUMN = "image_id"
LABEL_COLUMN = "label"
LABEL_RAW_COLUMN = "label_raw"

LABEL_IS_NORMALIZED_0_1 = True
SCORE_RANGE = (1.0, 10.0)
LABEL_RAW_SCALE = 10.0
LABEL_MIN = 1.0

ACC_THRESH_05 = 0.05
ACC_THRESH_10 = 0.10

def normalize_score(score, score_range=None):
    if not LABEL_IS_NORMALIZED_0_1:
        return score
    if score_range is None:
        score_range = SCORE_RANGE
    smin, smax = score_range
    return (score - smin) / (smax - smin + 1e-12)

def denormalize_score(score, score_range=None):
    if not LABEL_IS_NORMALIZED_0_1:
        return score
    if score_range is None:
        score_range = SCORE_RANGE
    smin, smax = score_range
    return score * (smax - smin) + smin

# ============================================================
# 特征 schema
# ============================================================
CSV_SUB_FEATS = ["valid", "area_ratio"]
REGION_META_FEATS = ["valid", "area_ratio"]

SEMANTIC_LABELS = {
    0: "background",
    1: "face",
    2: "sky",
    3: "building",
    4: "vegetation",
    5: "water",
    6: "object",
    7: "noise",
}

CORE_FEATURE_ORDER = [
    "edge_sharpness",
    "nr_sharpness",
    "color_cast",
    "saturation",
    "color_balance",
    "over_exposure",
    "under_exposure",
    "brightness_uniformity",
    "composition",
    "symmetry",
    "texture",
    "contrast",
]

GLOBAL_FEATURE_NAMES = list(CORE_FEATURE_ORDER)
REGION_FEATURE_NAMES = list(CORE_FEATURE_ORDER) + list(CSV_SUB_FEATS)
HANDCRAFTED_GLOBAL_FEATURE_NAMES = list(GLOBAL_FEATURE_NAMES)
HANDCRAFTED_REGION_FEATURE_NAMES = list(REGION_FEATURE_NAMES)
REGION_NAMES = [SEMANTIC_LABELS[i] for i in sorted(SEMANTIC_LABELS.keys())]

# ============================================================
# 特征维度
# ============================================================
SEG_NUM_CLASSES = 8
REGION_COUNT = SEG_NUM_CLASSES

GLOBAL_CORE_DIM = len(CORE_FEATURE_ORDER)
REGION_CORE_DIM = len(CORE_FEATURE_ORDER)
REGION_META_PER_REGION = len(CSV_SUB_FEATS)
REGION_META_DIM = REGION_COUNT * REGION_META_PER_REGION

GLOBAL_FEATURE_DIM = GLOBAL_CORE_DIM
TOTAL_FEATURE_DIM = GLOBAL_CORE_DIM + REGION_COUNT * REGION_CORE_DIM

GLOBAL_HANDCRAFTED_DIM = GLOBAL_CORE_DIM
REGION_HANDCRAFTED_DIM = REGION_CORE_DIM + REGION_META_PER_REGION

MANUAL_FEATURE_DIM = 0

# ============================================================
# 模型维度 / 分支相关
# ============================================================
REGION_FEATURE_DIM = 512
FUSED_GLOBAL_DIM = 1024

GLOBAL_HAND_EMBED_DIM = 64
REGION_HAND_EMBED_DIM = 32
GLOBAL_SCORE_HIDDEN_DIM = 1024
REGION_SCORE_HIDDEN_DIM = 512
REGION_TYPE_EMBED_DIM = 16

MIN_REGION_MASS = 10.0

# ============================================================
# RoIAlign 区域深特征
# ============================================================
USE_ROI_ALIGN_REGION_FEATURES = True
ROI_OUTPUT_SIZE = 7
ROI_SAMPLING_RATIO = 2
ROI_ALIGN_BINARIZE_THRESHOLD = 0.5
ROI_MIN_BOX_SIZE = 1.0

# ============================================================
# 注意力 / Dropout
# ============================================================
ATTENTION_HEADS = 8
ATTENTION_DROPOUT = 0.25
PREDICTION_DROPOUT = 0.5
REGION_DROPOUT_PROB = 0.4

# ============================================================
# 数据增强（更稳的正则化）
# ============================================================
USE_TRAIN_AUG = True
AUG_ENABLE_HFLIP = True
AUG_HFLIP_PROB = 0.5
AUG_COLOR_JITTER_PROB = 0.6
AUG_BRIGHTNESS = 0.10
AUG_CONTRAST = 0.10
AUG_SATURATION = 0.08
AUG_HUE = 0.02
AUG_GAUSSIAN_BLUR_PROB = 0.20
AUG_GAUSSIAN_BLUR_RADIUS = (0.1, 1.0)
AUG_GRAYSCALE_PROB = 0.05

# ============================================================
# 训练总控
# ============================================================
EPOCHS = 30
WARMUP_EPOCHS = 3
PHASE2_EPOCHS = 3
PHASE3_EPOCHS = 4

BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()
EARLY_STOP_PATIENCE = 2
GRADIENT_CLIP = 1.0
SAVE_BEST_ONLY = True

# ============================================================
# 优化器 / 调度器（更保守的 phase2/phase3）
# ============================================================
LR = 8e-5
WEIGHT_DECAY = 5.0e-2
BACKBONE_WEIGHT_DECAY = 5.0e-2

SCHEDULER_TYPE = "cosine"
MIN_LR = 1e-6
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 1
PLATEAU_THRESHOLD = 1e-4

WARMUP_LR_SCALE = 0.05
RESNET_LR_SCALE = 0.05
VIT_LR_SCALE = 0.1
SEG_LR_SCALE = 0.0

PHASE1_HEAD_LR = LR * WARMUP_LR_SCALE

# 防止 phase2 第 2 个 epoch 就明显过拟合
PHASE2_HEAD_LR = 3.0e-5
PHASE2_VIT_LR = 4.0e-6
PHASE2_RESNET_LR = 1.0e-6

PHASE3_HEAD_LR = 1.0e-5
PHASE3_VIT_LR = 1.0e-6
PHASE3_RESNET_LR = 2.0e-6

PHASE2_RESNET_UNFREEZE_MODE = "layer4"
RESNET_UNFREEZE_MODE = "layer4"
VIT_UNFREEZE_START_LAYER = 10
PHASE3_ONLY_IF_PHASE2_IMPROVED = True

# ============================================================
# 损失函数
# ============================================================
GLOBAL_LOSS_TYPE = "smooth_l1"
GLOBAL_LOSS_BETA = 0.05
LOSS_GLOBAL_WEIGHT = 1.0

# 这些作为 phase2/3 的终点权重
LOSS_ALIGNMENT_WEIGHT = 0.03
LOSS_CONSISTENCY_WEIGHT = 0.02
LOSS_REGION_QUALITY_WEIGHT = 0.03
LOSS_RANKING_WEIGHT = 0.01

# 逐步升权重，避免 phase2 一开始就把区域损失拉太重
PHASE2_ALIGNMENT_START = 0.01
PHASE2_ALIGNMENT_END = 0.03
PHASE2_CONSISTENCY_START = 0.01
PHASE2_CONSISTENCY_END = 0.02
PHASE2_REGION_QUALITY_START = 0.01
PHASE2_REGION_QUALITY_END = 0.03
PHASE2_RANKING_START = 0.00
PHASE2_RANKING_END = 0.01

PHASE3_ALIGNMENT_START = 0.03
PHASE3_ALIGNMENT_END = 0.05
PHASE3_CONSISTENCY_START = 0.02
PHASE3_CONSISTENCY_END = 0.03
PHASE3_REGION_QUALITY_START = 0.03
PHASE3_REGION_QUALITY_END = 0.05
PHASE3_RANKING_START = 0.01
PHASE3_RANKING_END = 0.01

RANKING_PAIR_THRESHOLD = 0.05
RANKING_MARGIN = 0.03

BEST_METRIC_SRCC_WEIGHT = 0.70
BEST_METRIC_PCC_WEIGHT = 0.30

REGION_PRESENCE_MIN_AREA_RATIO = 0.002
ROI_ADAPTIVE_THRESHOLD_RATIO = 0.5

# ============================================================
# 其他
# ============================================================
SHAP_BACKGROUND_SAMPLES = 100

# ============================================================
# ModelConfig：必须和 full_model.py 对齐
# ============================================================
class ModelConfig:
    seg_model_name_or_path = SEGFORMER_BASE_DIR
    seg_finetuned_ckpt = SEGFORMER_FINETUNE_CKPT
    resnet50_ckpt = RESNET50_CKPT

    vit_model_name_or_path = VIT_BASE_CKPT
    vit_ckpt_path = VIT_BASE_CKPT

    num_regions = SEG_NUM_CLASSES
    region_feature_dim = REGION_FEATURE_DIM
    global_feature_dim = FUSED_GLOBAL_DIM
    handcrafted_global_dim = GLOBAL_HANDCRAFTED_DIM
    handcrafted_region_dim = REGION_HANDCRAFTED_DIM

    global_hand_embed_dim = GLOBAL_HAND_EMBED_DIM
    region_hand_embed_dim = REGION_HAND_EMBED_DIM
    global_score_hidden_dim = GLOBAL_SCORE_HIDDEN_DIM
    region_score_hidden_dim = REGION_SCORE_HIDDEN_DIM
    region_type_embed_dim = REGION_TYPE_EMBED_DIM

    cross_attn_heads = ATTENTION_HEADS
    cross_attn_dropout = ATTENTION_DROPOUT
    region_dropout_prob = REGION_DROPOUT_PROB

    score_range = SCORE_RANGE
    min_region_mass = MIN_REGION_MASS

    use_roi_align_region_features = USE_ROI_ALIGN_REGION_FEATURES
    roi_output_size = ROI_OUTPUT_SIZE
    roi_sampling_ratio = ROI_SAMPLING_RATIO
    roi_align_binarize_threshold = ROI_ALIGN_BINARIZE_THRESHOLD
    roi_min_box_size = ROI_MIN_BOX_SIZE

    freeze_seg = SEG_FREEZE
    freeze_vit = FREEZE_VIT
    use_online_handcrafted_extractor = USE_ONLINE_HANDCRAFTED_EXTRACTOR
    disable_regions = False
    ablate_segmentation = False
    ablate_cross_attention = False

    global_feature_names = GLOBAL_FEATURE_NAMES
    region_feature_names = REGION_FEATURE_NAMES
    region_names = REGION_NAMES

    # ============================================================
    # 消融实验配置
    # ============================================================
    # Grid Regions: 使用固定网格区域代替语义分割区域
    use_grid_regions = False
    grid_rows = 2  # 网格行数
    grid_cols = 4  # 网格列数 (2×4=8 regions，与语义类别数量一致)
    
    # Concat Fusion: 使用拼接融合代替交叉注意力融合
    use_concat_fusion = False
    
    # Remove Statistical Features: 移除统计美学特征
    remove_statistical_features = False

# ============================================================
# TrainConfig
# ============================================================
class TrainConfig:
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    ACCUMULATION_STEPS = ACCUMULATION_STEPS

    lr = LR
    warmup_lr_scale = WARMUP_LR_SCALE
    resnet_lr_scale = RESNET_LR_SCALE
    vit_lr_scale = VIT_LR_SCALE
    seg_lr_scale = SEG_LR_SCALE

    weight_decay = WEIGHT_DECAY
    grad_clip = GRADIENT_CLIP
    scheduler = SCHEDULER_TYPE
    warmup_epochs = WARMUP_EPOCHS
    min_lr = MIN_LR

    use_amp = True
    seed = SEED
    early_stop_patience = EARLY_STOP_PATIENCE

    num_workers = NUM_WORKERS
    pin_memory = PIN_MEMORY

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
