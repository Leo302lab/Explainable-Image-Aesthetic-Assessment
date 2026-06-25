import torch
import numpy as np
import sys
from typing import Dict  # 添加这行
# 添加项目根目录到Python路径
project_root = "/root/autodl-tmp"
sys.path.insert(0, project_root)
from scipy.stats import pearsonr
from utils.config import DEVICE


def mse_loss(pred, target):
    """均方误差损失"""
    return torch.nn.MSELoss()(pred, target)


def mae_loss(pred, target):
    """平均绝对误差损失"""
    return torch.nn.L1Loss()(pred, target)


def pearson_correlation(pred, target):
    """皮尔逊相关系数（评估预测与真实评分的相关性）"""
    pred_np = pred.cpu().detach().numpy().flatten()
    target_np = target.cpu().detach().numpy().flatten()
    corr, _ = pearsonr(pred_np, target_np)
    return corr if not np.isnan(corr) else 0.0


def region_score_accuracy(region_preds, region_targets, threshold=1.0):
    """区域评分准确率：预测与真实评分差值≤threshold的比例"""
    diff = torch.abs(region_preds - region_targets)
    acc = (diff <= threshold).float().mean()
    return acc.item()


def defect_localization_accuracy(defect_pred_regions, true_defect_regions):
    """缺陷定位准确率：基于区域交集计算（IoU逻辑）
    Args:
        defect_pred_regions: 预测的缺陷区域列表（每个元素为区域ID列表）
        true_defect_regions: 真实的缺陷区域列表（每个元素为区域ID列表）
    Returns:
        precision: 精确率（预测为缺陷且实际为缺陷的比例）
        recall: 召回率（实际为缺陷且被预测的比例）
        f1: F1分数
    """
    if len(defect_pred_regions) == 0 or len(true_defect_regions) == 0:
        return 0.0, 0.0, 0.0

    # 转换为集合便于计算
    pred_set = set(sum(defect_pred_regions, []))  # 展平所有预测缺陷区域
    true_set = set(sum(true_defect_regions, []))  # 展平所有真实缺陷区域

    # 计算交集和并集
    tp = len(pred_set & true_set)  # 真正例：预测正确的缺陷区域
    fp = len(pred_set - true_set)  # 假正例：预测错误的缺陷区域
    fn = len(true_set - pred_set)  # 假负例：未预测到的缺陷区域

    # 计算指标（避免除零）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def explanation_accuracy(pred_explanations, true_explanations):
    """解释准确率：人工评估解释文本与真实原因的匹配度（0-1分）
    注：需人工标注真实解释，计算预测解释的平均得分
    """
    if len(pred_explanations) != len(true_explanations):
        raise ValueError("预测解释与真实解释数量不一致")

    # 假设人工标注每个解释的匹配得分（0=完全不匹配，1=完全匹配）
    # 此处简化为计算关键词匹配率（实际需人工评估）
    def keyword_match(pred_exp, true_exp):
        pred_keywords = set(pred_exp.lower().split())
        true_keywords = set(true_exp.lower().split())
        match_ratio = len(pred_keywords & true_keywords) / len(true_keywords) if true_keywords else 0.0
        return match_ratio

    avg_accuracy = np.mean([keyword_match(p, t) for p, t in zip(pred_explanations, true_explanations)])
    return avg_accuracy


# 指标集合（训练/评估时调用）
METRICS = {
    "mse": mse_loss,
    "mae": mae_loss,
    "pearson": pearson_correlation,
    "region_acc": region_score_accuracy,
    "defect_precision": lambda p, t: defect_localization_accuracy(p, t)[0],
    "defect_recall": lambda p, t: defect_localization_accuracy(p, t)[1],
    "defect_f1": lambda p, t: defect_localization_accuracy(p, t)[2],
    "explanation_acc": explanation_accuracy
}


def calculate_segmentation_metrics(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 8) -> Dict[
    str, float]:
    """
    计算语义分割的评估指标
    Args:
        predictions: 预测的分割掩码 (B, H, W) 或 (B, C, H, W)
        targets: 真实的分割掩码 (B, H, W)
        num_classes: 类别数量
    Returns:
        分割指标字典
    """
    if predictions.dim() == 4:  # (B, C, H, W) 格式
        # 取最大概率的类别
        predictions = torch.argmax(predictions, dim=1)

    # 确保数据类型一致
    predictions = predictions.long()
    targets = targets.long()

    # 初始化指标
    total_pixels = predictions.numel()
    correct_pixels = (predictions == targets).sum().item()
    accuracy = correct_pixels / total_pixels

    # 计算每个类别的IoU
    ious = []
    for class_id in range(num_classes):
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        ious.append(iou)

    # 平均IoU（忽略背景类0）
    mean_iou = np.mean(ious[1:]) if len(ious) > 1 else ious[0]

    # 计算Dice系数
    dice_scores = []
    for class_id in range(1, num_classes):  # 跳过背景
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)

        intersection = (pred_mask & target_mask).sum().item()
        dice = (2.0 * intersection) / (pred_mask.sum().item() + target_mask.sum().item() + 1e-8)
        dice_scores.append(dice)

    mean_dice = np.mean(dice_scores) if dice_scores else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_dice": round(mean_dice, 4),
        "per_class_iou": {f"class_{i}": round(iou, 4) for i, iou in enumerate(ious)}
    }


def calculate_segmentation_loss(predictions: torch.Tensor, targets: torch.Tensor,
                                loss_type: str = "cross_entropy") -> torch.Tensor:
    """
    计算语义分割损失
    Args:
        predictions: 预测的分割logits (B, C, H, W)
        targets: 真实的分割掩码 (B, H, W)
        loss_type: 损失类型 ("cross_entropy", "dice", "focal")
    Returns:
        分割损失
    """
    if loss_type == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        loss = criterion(predictions, targets)
    elif loss_type == "dice":
        # Dice损失实现
        from torch import einsum
        batch_size = predictions.size(0)
        num_classes = predictions.size(1)

        # 将预测转换为概率
        probs = torch.softmax(predictions, dim=1)

        # 创建one-hot编码的目标
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # 计算Dice系数
        intersection = einsum('bcwh,bcwh->bc', probs, targets_one_hot)
        union = einsum('bcwh->bc', probs) + einsum('bcwh->bc', targets_one_hot)
        dice = (2. * intersection + 1e-8) / (union + 1e-8)
        loss = 1 - dice.mean()
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")

    return loss
