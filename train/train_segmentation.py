import os

# 关键修复：强制关闭 weights_only 检查 (针对 PyTorch 2.5/2.6+)
os.environ["TORCH_WEIGHTS_ONLY"] = "0"
import traceback
import sys
import random
import cv2
import numpy as np
import torch
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from tqdm import tqdm
from transformers import SegformerImageProcessor
from segformer_module import SegmentationBranch
# --- 引入混合精度所需库 ---
from torch.cuda.amp import autocast as cuda_autocast, GradScaler

# 兼容新旧版本的 autocast
def autocast(device_type="cuda"):
    return cuda_autocast()

# --- 路径与环境配置（保持不动） ---
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

BASE_DIR = "/root/autodl-tmp"
IMAGE_DIR = os.path.join(BASE_DIR, "Data_preprocess/cleaned")
MASK_DIR = os.path.join(BASE_DIR, "Data_preprocess/semantic_masks")
SPLIT_DIR = os.path.join(BASE_DIR, "AVA＿dataset/split_files")
Model_DIR = os.path.join(BASE_DIR, "pythonProject2/training/models/nvidia/segformer-b0-finetuned-ade-512-512")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints/segmentation")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================= 1. 优化后的参数 (针对 RTX 3090) =================
NUM_CLASSES = 8
BATCH_SIZE = 32  # 3090 显存充足，使用 32 提升梯度稳定性
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4  # 降低学习率，减少震荡
WEIGHT_DECAY = 0.01  # 增强正则化，提高论文中的泛化性指标
WARMUP_EPOCHS = 3  # 缩短 Warmup 周期
PATIENCE = 8  # 调整早停耐心
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 3. 数据增强函数（保持不动） =================
class DataAugmentor:
    def random_crop(self, image, mask, crop_size=450):
        h, w = image.shape[:2]
        if h > crop_size and w > crop_size:
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)
            image = image[top:top + crop_size, left:left + crop_size]
            mask = mask[top:top + crop_size, left:left + crop_size]
            image = cv2.resize(image, (w, h))
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return image, mask

    def random_flip(self, image, mask):
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return image, mask

    def adjust_brightness(self, image):
        brightness_factor = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        return image

    def augment(self, image, mask, split="train"):
        if split == "train":
            image, mask = self.random_flip(image, mask)
            image, mask = self.random_crop(image, mask)
            image = self.adjust_brightness(image)
        return image, mask


# ================= 4. 数据集定义（保持不动） =================
class SegmentationDataset(Dataset):
    def __init__(self, split, processor):
        self.split = split
        self.processor = processor
        self.augmentor = DataAugmentor()

        split_file = os.path.join(SPLIT_DIR, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"找不到划分文件: {split_file}")

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        self.valid_ids = []
        print(f"[{split}] 正在校验文件路径...")
        for img_id in self.image_ids:
            img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
            mask_path = os.path.join(MASK_DIR, f"{img_id}.png")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.valid_ids.append(img_id)

        print(f"[{split}] 最终有效样本: {len(self.valid_ids)} / {len(self.image_ids)}")
        if len(self.valid_ids) == 0:
            raise RuntimeError(f"❌ {split} 集有效样本为 0！")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        image = cv2.cvtColor(cv2.imread(os.path.join(IMAGE_DIR, f"{img_id}.jpg")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(MASK_DIR, f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
        image, mask = self.augmentor.augment(image, mask, self.split)

        encoded = self.processor(image, mask, return_tensors="pt")
        for k, v in encoded.items():
            encoded[k] = v.squeeze()
        return encoded


# ================= 5. 辅助函数 =================
def calculate_dice(mask_pred, labels, num_classes):
    """计算多类 Dice 系数"""
    dice = 0.0
    for class_idx in range(num_classes):
        pred = (mask_pred == class_idx).float()
        true = (labels == class_idx).float()
        intersection = (pred * true).sum()
        union = pred.sum() + true.sum()
        if union > 0:
            dice += (2.0 * intersection) / union
    return dice / num_classes


def calculate_miou(mask_pred, labels, num_classes):
    """计算 mIoU（Mean Intersection over Union）"""
    iou_sum = 0.0
    for class_idx in range(num_classes):
        pred = (mask_pred == class_idx).float()
        true = (labels == class_idx).float()
        intersection = (pred * true).sum()
        union = pred.sum() + true.sum() - intersection
        if union > 0:
            iou_sum += intersection / union
    return iou_sum / num_classes


def print_final_metrics(best_metrics, dataset_name="验证集"):
    """打印最终指标（保留四位小数）"""
    print(f"\n{'='*60}")
    print(f"📊 {dataset_name} 评估结果")
    print(f"{'='*60}")
    print(f"  mIoU          : {best_metrics['miou']:.4f}")
    print(f"  Mean Dice     : {best_metrics['dice']:.4f}")
    print(f"  Pixel Accuracy: {best_metrics['acc']*100:.2f}%")
    print(f"{'='*60}\n")


def find_model_path(search_dirs=None):
    """
    自动搜索包含模型文件的目录（支持 pytorch_model.bin 和 model.safetensors 两种格式）
    Args:
        search_dirs: 要搜索的目录列表，默认为项目根目录下的 models 文件夹
    Returns:
        包含模型文件的目录路径，如果未找到返回 None
    """
    # 支持的模型文件格式
    target_files = ["pytorch_model.bin", "model.safetensors"]
    
    if search_dirs is None:
        search_dirs = [
            os.path.join(project_root, "models"),
            os.path.join(current_dir, "models"),
            os.path.join(BASE_DIR, "models"),
            os.path.join(project_root, "training", "models"),
            Model_DIR,  # 也搜索原来指定的路径
        ]

    print(f"🔍 正在搜索模型文件...")
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for root, dirs, files in os.walk(search_dir):
            for target_file in target_files:
                if target_file in files:
                    print(f"✅ 找到模型文件 '{target_file}' 在: {root}")
                    return root
    
    print(f"❌ 未找到包含以下模型文件的目录: {target_files}")
    return None


def evaluate_on_test_set(model, test_loader, num_classes, device):
    """在测试集上评估模型"""
    model.eval()
    test_loss, test_acc, test_dice, test_miou = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with cuda_autocast():
                outputs = model(x=pixel_values, target_mask=labels)
                test_loss += outputs["loss"].item() * pixel_values.shape[0]
                mask_pred = outputs["logits"].argmax(dim=1)
                test_acc += (mask_pred == labels).float().mean().item() * pixel_values.shape[0]
                test_dice += calculate_dice(mask_pred, labels, num_classes) * pixel_values.shape[0]
                test_miou += calculate_miou(mask_pred, labels, num_classes) * pixel_values.shape[0]

    return {
        "loss": test_loss / len(test_loader.dataset),
        "acc": test_acc / len(test_loader.dataset),
        "dice": test_dice / len(test_loader.dataset),
        "miou": test_miou / len(test_loader.dataset)
    }


# ================= 6. 训练主流程 =================
def train():
    print("🚀 启动 RTX 3090 优化版语义分割微调训练...")

    # 1. 初始化模型
    seg_module = SegmentationBranch(pretrained=True, freeze_backbone=False)
    model = seg_module.to(DEVICE)

    # 2. 自动查找模型路径
    model_path = find_model_path()
    if model_path is None:
        print("❌ 未找到模型文件！请确保 pytorch_model.bin 或 model.safetensors 存在于项目中")
        raise FileNotFoundError("模型文件未找到")
    
    # 3. 加载 Processor
    try:
        processor = SegformerImageProcessor.from_pretrained(model_path, local_files_only=True)
        print(f"✅ 从本地路径加载 Processor 成功: {model_path}")
    except Exception as e:
        print(f"⚠️ 本地加载失败，使用默认配置: {e}")
        processor = SegformerImageProcessor(
            size={'height': 512, 'width': 512},
            do_resize=True, do_normalize=True, do_pad=False,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]
        )

    # 3. 准备数据
    train_dataset = SegmentationDataset("train", processor)
    val_dataset = SegmentationDataset("val", processor)
    test_dataset = SegmentationDataset("test", processor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # 4. 优化器（应用权重衰减）
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 混合精度缩放器
    scaler = GradScaler()

    # 使用 CosineAnnealingWarmRestarts 调度器，更稳定
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,        # 每5个epoch重启一次
        T_mult=2,     # 重启后周期翻倍
        eta_min=1e-6  # 最小学习率
    )

    best_acc = 0.0
    best_metrics = {"miou": 0.0, "dice": 0.0, "acc": 0.0}
    early_stop_counter = 0

    # 5. 训练循环
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            # --- 核心逻辑：开启混合精度上下文 ---
            with cuda_autocast():
                outputs = model(x=pixel_values, target_mask=labels)
                loss = outputs["loss"]

            # --- 核心逻辑：反向传播缩放 ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * pixel_values.shape[0]
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # 验证
        model.eval()
        val_loss, val_acc, val_dice, val_miou = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                with cuda_autocast():
                    outputs = model(x=pixel_values, target_mask=labels)
                    val_loss += outputs["loss"].item() * pixel_values.shape[0]
                    mask_pred = outputs["logits"].argmax(dim=1)
                    val_acc += (mask_pred == labels).float().mean().item() * pixel_values.shape[0]
                    val_dice += calculate_dice(mask_pred, labels, NUM_CLASSES) * pixel_values.shape[0]
                    val_miou += calculate_miou(mask_pred, labels, NUM_CLASSES) * pixel_values.shape[0]

        train_avg_loss = train_loss / len(train_loader.dataset)
        val_avg_loss = val_loss / len(val_loader.dataset)
        val_avg_acc = val_acc / len(val_loader.dataset)
        val_avg_dice = val_dice / len(val_loader.dataset)
        val_avg_miou = val_miou / len(val_loader.dataset)

        # 学习率调度
        scheduler.step(epoch)
        print(f"   [Scheduler] 学习率: {optimizer.param_groups[0]['lr']:.8f}")

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1} 总结: Loss={train_avg_loss:.4f}, Val_Loss={val_avg_loss:.4f}, "
            f"mIoU={val_avg_miou:.4f}, Dice={val_avg_dice:.4f}, Acc={val_avg_acc*100:.2f}%, Time={epoch_time:.1f}s")

        # 保存与早停
        if val_avg_dice > best_acc:
            best_acc = val_avg_dice
            best_metrics = {"miou": val_avg_miou, "dice": val_avg_dice, "acc": val_avg_acc}
            save_path = os.path.join(CHECKPOINT_DIR, "best_segformer_finetune_V618.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🌟 性能提升，模型已保存: {save_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"🛑 触发早停")
                break

    # 训练结束后打印验证集最终指标
    print_final_metrics(best_metrics, "验证集")

    # 在测试集上进行最终评估
    print("\n🔍 正在测试集上评估最佳模型...")
    best_model = SegmentationBranch(pretrained=False)
    best_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_segformer_finetune_V618.pth")))
    best_model.to(DEVICE)

    test_metrics = evaluate_on_test_set(best_model, test_loader, NUM_CLASSES, DEVICE)
    print_final_metrics(test_metrics, "测试集")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n❌ 训练异常中断: {e}")
        traceback.print_exc()
