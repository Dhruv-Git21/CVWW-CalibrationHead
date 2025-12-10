"""
Evaluation script for CIFAR-100 classification models.

Usage:
    python src/eval_cls.py --config configs/cifar100_resnet50.yaml --checkpoint runs/cifar_resnet50/best_model.pth
"""
import argparse
import yaml
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.cifar100 import CIFAR100DataModule
from src.models.resnet import build_resnet_model
from src.models.vit import build_vit_model
from src.utils.common import seed_everything, load_checkpoint, get_device
from src.utils.train_utils import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-100 classification model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--save-confusion', action='store_true', help='Save confusion matrix')
    return parser.parse_args()


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config):
    """Build model based on config."""
    model_config = config['model']
    model_name = model_config['name']
    num_classes = model_config['num_classes']
    
    if model_name in ['resnet50', 'wide_resnet28_10']:
        model = build_resnet_model(model_name, num_classes=num_classes, **model_config)
    elif model_name in ['vit_tiny', 'deit_tiny']:
        model = build_vit_model(model_name, num_classes=num_classes, **model_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


@torch.no_grad()
def get_predictions(model, dataloader, device):
    """Get all predictions and targets."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())
        all_probs.append(probs.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    
    return all_preds, all_targets, all_probs


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(20, 18))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot using seaborn
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def compute_per_class_accuracy(cm):
    """Compute per-class accuracy from confusion matrix."""
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    return per_class_acc


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Set random seed
    seed = config.get('seed', 42)
    seed_everything(seed, deterministic=False)  # Deterministic not needed for eval
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Setup data
    print("\n=== Setting up data ===")
    data_config = config.get('data', {})
    eval_config = config.get('eval', {})
    
    datamodule = CIFAR100DataModule(
        data_dir=config['paths'].get('data_dir', './data'),
        batch_size=eval_config.get('batch_size', 256),
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        val_split=0.1,
        use_randaugment=False,  # No augmentation for eval
    )
    
    datamodule.prepare_data()
    datamodule.setup()
    
    # Get appropriate dataloader
    if args.split == 'val':
        dataloader = datamodule.val_dataloader()
        print(f"Evaluating on validation set ({len(datamodule.val_dataset)} samples)")
    else:
        dataloader = datamodule.test_dataloader()
        print(f"Evaluating on test set ({len(datamodule.test_dataset)} samples)")
    
    # Build model
    print("\n=== Building model ===")
    model = build_model(config)
    model = model.to(device)
    print(f"Model: {config['model']['name']}")
    
    # Load checkpoint
    print(f"\n=== Loading checkpoint: {args.checkpoint} ===")
    load_checkpoint(args.checkpoint, model, device=device)
    
    # Evaluate
    print("\n=== Evaluating ===")
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        desc='Eval',
    )
    
    print(f"\n=== Results ===")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Top-1 Accuracy: {metrics['acc1']:.2f}%")
    print(f"Top-5 Accuracy: {metrics['acc5']:.2f}%")
    
    # Get predictions for detailed analysis
    print("\n=== Computing detailed metrics ===")
    preds, targets, probs = get_predictions(model, dataloader, device)
    
    # Compute confusion matrix
    cm = confusion_matrix(targets, preds)
    
    # Per-class accuracy
    per_class_acc = compute_per_class_accuracy(cm)
    class_names = CIFAR100DataModule.get_class_names()
    
    print(f"\nPer-class accuracy (mean): {per_class_acc.mean():.4f}")
    print(f"Per-class accuracy (std): {per_class_acc.std():.4f}")
    print(f"Per-class accuracy (min): {per_class_acc.min():.4f}")
    print(f"Per-class accuracy (max): {per_class_acc.max():.4f}")
    
    # Find best and worst classes
    best_idx = np.argsort(per_class_acc)[-5:][::-1]
    worst_idx = np.argsort(per_class_acc)[:5]
    
    print("\nTop 5 best performing classes:")
    for idx in best_idx:
        print(f"  {class_names[idx]}: {per_class_acc[idx]:.4f}")
    
    print("\nTop 5 worst performing classes:")
    for idx in worst_idx:
        print(f"  {class_names[idx]}: {per_class_acc[idx]:.4f}")
    
    # Save confusion matrix
    if args.save_confusion:
        output_dir = Path(config['paths']['output_dir'])
        cm_path = output_dir / f'confusion_matrix_{args.split}.png'
        plot_confusion_matrix(cm, class_names, cm_path)
        
        # Save per-class accuracies
        acc_path = output_dir / f'per_class_accuracy_{args.split}.txt'
        with open(acc_path, 'w') as f:
            f.write("Class,Accuracy\n")
            for name, acc in zip(class_names, per_class_acc):
                f.write(f"{name},{acc:.4f}\n")
        print(f"Per-class accuracies saved to {acc_path}")
    
    # Classification report
    print("\n=== Classification Report (Sample) ===")
    # Print report for first 10 classes to avoid clutter
    sample_classes = list(range(10))
    sample_names = [class_names[i] for i in sample_classes]
    sample_targets = np.where(np.isin(targets, sample_classes))[0]
    
    if len(sample_targets) > 0:
        report = classification_report(
            targets[sample_targets],
            preds[sample_targets],
            labels=sample_classes,
            target_names=sample_names,
            digits=4
        )
        print(report)
    
    print("\n=== Evaluation complete ===")


if __name__ == '__main__':
    main()
