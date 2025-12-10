"""
Training utilities for classification models.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from .common import AverageMeter, accuracy


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross entropy loss.
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = nn.functional.log_softmax(pred, dim=1)
        loss = -(smooth_one_hot * log_prob).sum(dim=1).mean()
        return loss


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    Apply Mixup augmentation.
    
    Args:
        x: Input images
        y: Labels
        alpha: Mixup parameter
    
    Returns:
        Mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    Apply CutMix augmentation.
    
    Args:
        x: Input images
        y: Labels
        alpha: CutMix parameter
    
    Returns:
        Mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get random box
    _, _, H, W = x.size()
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform sample
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    clip_grad: float = 0.0,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        scaler: AMP scaler
        clip_grad: Gradient clipping value (0 = no clipping)
        mixup_alpha: Mixup alpha (0 = no mixup)
        cutmix_alpha: CutMix alpha (0 = no cutmix)
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        batch_size = images.size(0)
        
        # Apply augmentation
        use_mixup = mixup_alpha > 0 and np.random.rand() < 0.5
        use_cutmix = cutmix_alpha > 0 and np.random.rand() < 0.5
        
        if use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
        elif use_cutmix:
            images, targets_a, targets_b, lam = cutmix_data(images, targets, cutmix_alpha)
        
        # Forward pass with AMP
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if use_mixup or use_cutmix:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            if use_mixup or use_cutmix:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        
        # Measure accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc@1': f'{top1.avg:.2f}',
            'acc@5': f'{top5.avg:.2f}'
        })
    
    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = 'Val',
) -> Dict[str, float]:
    """
    Evaluate model.
    
    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device
        desc: Description for progress bar
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    pbar = tqdm(dataloader, desc=f'[{desc}]')
    
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        batch_size = images.size(0)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Measure accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc@1': f'{top1.avg:.2f}',
            'acc@5': f'{top5.avg:.2f}'
        })
    
    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg,
    }


import numpy as np
