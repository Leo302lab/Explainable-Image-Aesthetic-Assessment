import os
import cv2
import torch
import numpy as np
import gc
from tqdm import tqdm
from torch.cuda.amp import autocast
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ================= 1. 环境与路径配置 =================
os.environ["TORCH_WEIGHTS_ONLY"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = "/root/autodl-tmp"
IMAGE_DIR = os.path.join(BASE_DIR, "Data_preprocess/cleaned")
NEW_MASK_DIR = os.path.join(BASE_DIR, "Data_preprocess/semantic_masks_finetuned")

# 已更新为 segformer_b0_finetune.pth
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints/segmentation/segformer_b0_finetune.pth")
MODEL_DIR = os.path.join(BASE_DIR, "pythonProject2/training/models/nvidia/segformer-b0-finetuned-ade-512-512")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(NEW_MASK_DIR, exist_ok=True)


def load_standard_model():
    print(f"🧠 正在加载标准 SegFormer 结构并自动对齐权重...")

    # 1. 初始化标准模型 (8类)
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_DIR,
        num_labels=8,
        ignore_mismatched_sizes=True,
        local_files_only=True
    )

    # 2. 加载更新后的 .pth 权重
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ 找不到权重文件: {CHECKPOINT_PATH}")

    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # 3. 清洗 Key (对齐 HuggingFace 标准结构)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("segformer.segformer.", "").replace("segformer.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE).eval()

    processor = SegformerImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    return model, processor


def run_extraction():
    model, processor = load_standard_model()
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"🚀 开始处理 {len(image_files)} 张图片 | 显卡: {torch.cuda.get_device_name(0)}")

    with torch.no_grad():
        for i, img_name in enumerate(tqdm(image_files, desc="重刷高质量掩码")):
            img_path = os.path.join(IMAGE_DIR, img_name)
            save_path = os.path.join(NEW_MASK_DIR, os.path.splitext(img_name)[0] + '.png')

            # 增量处理：如果文件已存在则跳过
            if os.path.exists(save_path): continue

            try:
                # 使用 CV2 读取，确保读取后的色彩空间转换正确
                image = cv2.imread(img_path)
                if image is None: continue

                h, w = image.shape[:2]
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 预处理
                inputs = processor(images=image_rgb, return_tensors="pt").to(DEVICE)

                # 推理 (使用通用的 autocast 以兼容不同版本的 PyTorch)
                with autocast():
                    outputs = model(**inputs)
                    logits = outputs.logits

                # 还原至原图尺寸
                upsampled_logits = torch.nn.functional.interpolate(
                    logits, size=(h, w), mode="bilinear", align_corners=False
                )
                mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

                # --- 核心优化：中值滤波 ---
                # 消除分割图中的孤立小点（椒盐噪声），这对后续 Cross-Attention 聚焦核心区域很有帮助
                mask = cv2.medianBlur(mask, 5)

                # 保存为单通道掩码图
                cv2.imwrite(save_path, mask)

                # --- 显存管理 ---
                if i % 500 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️ 图片 {img_name} 处理出错: {e}")
                continue

    print(f"\n🎉 掩码提取任务顺利完成！")
    print(f"📂 结果路径: {NEW_MASK_DIR}")


if __name__ == "__main__":
    run_extraction()
