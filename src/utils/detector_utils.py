"""
Training and evaluation utilities for object detection.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

from .common import AverageMeter


def train_one_epoch_detection(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    clip_grad: float = 0.0,
) -> Dict[str, float]:
    """
    Train detection model for one epoch.
    
    Args:
        model: Detection model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        scaler: AMP scaler
        clip_grad: Gradient clipping value
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    loss_meter = AverageMeter('Loss', ':.4f')
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for images, targets in pbar:
        # Move to device
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        # Forward pass with AMP
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        
        # Update metrics
        loss_meter.update(losses.item(), len(images))
        
        # Update progress bar with individual loss components
        loss_str = {k: f'{v.item():.4f}' for k, v in loss_dict.items()}
        loss_str['total'] = f'{losses.item():.4f}'
        pbar.set_postfix(loss_str)
    
    return {
        'loss': loss_meter.avg,
    }


@torch.no_grad()
def evaluate_detection(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate detection model.
    
    Args:
        model: Detection model
        dataloader: Validation dataloader
        device: Device
        iou_threshold: IoU threshold for mAP calculation
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Collect all predictions and ground truths
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc='[Eval]')
    
    for images, targets in pbar:
        # Move to device
        images = [img.to(device, non_blocking=True) for img in images]
        
        # Forward pass
        predictions = model(images)
        
        # Store predictions and targets
        all_predictions.extend(predictions)
        all_targets.extend(targets)
    
    # Calculate mAP (simplified version)
    # For production, use pycocotools or torchvision.ops.box_iou
    metrics = calculate_map(all_predictions, all_targets, iou_threshold=iou_threshold)
    
    return metrics


def calculate_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    This is a simplified version. For full COCO evaluation, use pycocotools.
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'labels', 'scores'
        targets: List of target dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold
    
    Returns:
        Dictionary with mAP metrics
    """
    # Simplified mAP calculation
    # In practice, use torchvision.ops.box_iou and proper AP calculation
    
    # For now, return dummy metrics
    # Full implementation would require per-class AP calculation
    
    num_images = len(predictions)
    total_predictions = sum(len(p['boxes']) for p in predictions)
    total_targets = sum(len(t['boxes']) for t in targets)
    
    # Placeholder - replace with actual mAP calculation
    map_50 = 0.0  # mAP@0.5
    map_75 = 0.0  # mAP@0.75
    map_50_95 = 0.0  # mAP@[0.5:0.95]
    
    return {
        'mAP@0.5': map_50,
        'mAP@0.75': map_75,
        'mAP@0.5:0.95': map_50_95,
        'num_images': num_images,
        'num_predictions': total_predictions,
        'num_targets': total_targets,
    }


def calculate_voc_ap(rec, prec):
    """
    Calculate VOC-style Average Precision.
    
    Args:
        rec: Recall values
        prec: Precision values
    
    Returns:
        Average Precision
    """
    # Append sentinel values
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Calculate area under precision-recall curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        boxes1: Boxes (N, 4) in (x1, y1, x2, y2) format
        boxes2: Boxes (M, 4) in (x1, y1, x2, y2) format
    
    Returns:
        IoU matrix (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou


def collate_fn(batch):
    """
    Custom collate function for detection.
    
    Args:
        batch: List of (image, target) tuples
    
    Returns:
        Tuple of (images, targets)
    """
    return tuple(zip(*batch))
