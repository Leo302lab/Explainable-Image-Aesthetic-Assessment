import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor
from segmentation_models_pytorch.losses import DiceLoss

# 环境配置：必须在 torch 加载前
os.environ["TORCH_WEIGHTS_ONLY"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 强制开启 HuggingFace 离线模式

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEMANTIC_CLASSES = 8


class SegmentationBranch(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()

        # ================= 1. 本地路径物理锁定 =================
        # 确保此路径下包含: config.json, pytorch_model.bin, preprocessor_config.json
        self.local_path = "/root/autodl-tmp/pythonProject2/training/models/nvidia/segformer-b0-finetuned-ade-512-512"

        if pretrained:
            print(f"📦 [SegmentationBranch] 正在执行强行本地加载: {self.local_path}")

            # 支持 pytorch_model.bin 和 model.safetensors 两种格式
            has_bin = os.path.exists(os.path.join(self.local_path, "pytorch_model.bin"))
            has_safetensors = os.path.exists(os.path.join(self.local_path, "model.safetensors"))
            
            if not has_bin and not has_safetensors:
                raise FileNotFoundError(f"❌ 关键文件缺失！在 {self.local_path} 未找到 pytorch_model.bin 或 model.safetensors")
            
            if has_safetensors:
                print(f"ℹ️ 检测到 Safetensors 格式模型文件")

            try:
                # 步骤 A: 先加载配置并修改类别数
                config = SegformerConfig.from_pretrained(self.local_path, local_files_only=True)
                config.num_labels = SEMANTIC_CLASSES

                # 步骤 B: 核心加载逻辑
                # 使用 from_pretrained 的同时配合 weights_only=False
                self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                    self.local_path,
                    config=config,
                    ignore_mismatched_sizes=True,
                    local_files_only=True,
                    weights_only=False
                )
                print("✅ [SegmentationBranch] 权重加载成功！")

            except Exception as e:
                print(f"⚠️ [SegmentationBranch] 常规加载失败，尝试手动注入 State Dict: {e}")
                self._manual_load_state_dict()
        else:
            print("🔄 [SegmentationBranch] 使用随机初始化")
            config = SegformerConfig(num_labels=SEMANTIC_CLASSES)
            self.segformer = SegformerForSemanticSegmentation(config)

        # 2. 冻结骨干网络 (Encoder)
        if freeze_backbone:
            for param in self.segformer.segformer.encoder.parameters():
                param.requires_grad = False
            print("🔒 SegFormer 编码器已冻结")

        # 3. 损失函数与权重
        # 频率: [背景, 人, 天, 建, 植, 水, 物, 扰]
        freqs = torch.tensor([15.56, 0.48, 5.48, 40.91, 15.39, 7.42, 14.76, 1.00])
        class_weights = (1.0 / (freqs / freqs.sum() + 1e-4))
        class_weights = (class_weights / class_weights.mean()).to(DEVICE)

        self.dice_loss = DiceLoss(mode="multiclass", classes=SEMANTIC_CLASSES, from_logits=True)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def _manual_load_state_dict(self):
        """手动解析并注入权重文件 (支持 pytorch_model.bin 和 model.safetensors)"""
        config = SegformerConfig.from_pretrained(self.local_path)
        config.num_labels = SEMANTIC_CLASSES
        self.segformer = SegformerForSemanticSegmentation(config)

        # 优先尝试 safetensors 格式
        bin_path = os.path.join(self.local_path, "pytorch_model.bin")
        safetensors_path = os.path.join(self.local_path, "model.safetensors")
        
        if os.path.exists(safetensors_path):
            print(f"ℹ️ 使用 Safetensors 格式加载权重")
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            # 这里的 map_location 非常重要，确保权重直接进入 CPU 内存再分发
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
        else:
            raise FileNotFoundError(f"❌ 未找到权重文件: {bin_path} 或 {safetensors_path}")

        # 必须过滤输出层：ADE20K(150) -> Our(8)
        # Segformer 的分类头通常叫 decode_head.classifier
        pop_keys = [k for k in state_dict.keys() if "classifier" in k]
        for k in pop_keys:
            state_dict.pop(k)

        msg = self.segformer.load_state_dict(state_dict, strict=False)
        print(f"✅ 手动注入成功，已自动跳过不匹配层: {msg.missing_keys[:2]}...")

    def forward(self, x, target_mask=None):
        # x: [B, 3, 512, 512]
        outputs = self.segformer(pixel_values=x)
        logits = outputs.logits  # [B, 8, 128, 128]

        # 线性插值上采样回原图尺寸
        upsampled_logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = None
        if target_mask is not None:
            target_mask = target_mask.long()
            loss = 0.5 * self.dice_loss(upsampled_logits, target_mask) + \
                   0.5 * self.ce_loss(upsampled_logits, target_mask)

        return {
            "logits": upsampled_logits,
            "mask_pred": upsampled_logits.argmax(dim=1),
            "loss": loss
        }


if __name__ == "__main__":
    try:
        model = SegmentationBranch(pretrained=True).to(DEVICE)
        dummy_input = torch.randn(1, 3, 512, 512).to(DEVICE)
        res = model(dummy_input)
        print(f"✅ 前向测试成功，输出 Logits 形状: {res['logits'].shape}")
    except:
        traceback.print_exc()
