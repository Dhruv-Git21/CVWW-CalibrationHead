"""
Common utilities: seed, checkpoints, accuracy, etc.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path


def seed_everything(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms (may impact performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for deterministic algorithms
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # Enable deterministic algorithms in PyTorch
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Compute top-k accuracy.
    
    Args:
        output: Model output logits (batch_size, num_classes)
        target: Ground truth labels (batch_size,)
        topk: Tuple of k values
    
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
):
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    
    # Save best model separately
    if is_best:
        best_path = str(Path(filepath).parent / 'best_model.pth')
        torch.save(state, best_path)
        print(f"Best model saved to {best_path}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to load checkpoint on
    
    Returns:
        Dictionary with loaded state information
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best metric: {checkpoint.get('best_metric', 'N/A')}")
    
    return checkpoint


def get_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """
    Compute confusion matrix.
    
    Args:
        preds: Predicted class indices (batch_size,)
        targets: Ground truth class indices (batch_size,)
        num_classes: Number of classes
    
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = '', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display progress during training."""
    
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device.
    
    Args:
        device: Device string ('cuda', 'cpu', or None for auto)
    
    Returns:
        torch.device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
