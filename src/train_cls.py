"""
Training script for CIFAR-100 classification models.

Usage:
    python src/train_cls.py --config configs/cifar100_resnet50.yaml
    python src/train_cls.py --config configs/cifar100_wrn2810.yaml
    python src/train_cls.py --config configs/cifar100_vit_tiny.yaml
"""
import argparse
import yaml
import os
import sys
from pathlib import Path
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.cifar100 import CIFAR100DataModule
from src.models.resnet import build_resnet_model
from src.models.vit import build_vit_model
from src.utils.common import (
    seed_everything, save_checkpoint, load_checkpoint,
    get_device, count_parameters
)
from src.utils.train_utils import (
    train_one_epoch, evaluate, LabelSmoothingCrossEntropy
)
from src.utils.metrics import MetricsLogger, print_metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR-100 classification model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
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
    
    # Build model (extract name from config to avoid duplicate kwarg)
    model_kwargs = {k: v for k, v in model_config.items() if k != 'name'}
    
    if model_name in ['resnet50', 'wide_resnet28_10']:
        model = build_resnet_model(model_name, **model_kwargs)
    elif model_name in ['vit_tiny', 'deit_tiny']:
        model = build_vit_model(model_name, **model_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def build_optimizer(model, config):
    """Build optimizer based on config."""
    train_config = config['train']
    optimizer_name = train_config['optimizer'].lower()
    lr = train_config['lr']
    weight_decay = train_config['weight_decay']
    
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=train_config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    """Build learning rate scheduler."""
    train_config = config['train']
    scheduler_name = train_config['scheduler']
    epochs = train_config['epochs']
    warmup_epochs = train_config.get('warmup_epochs', 0)
    
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=0,
        )
    elif scheduler_name == 'step':
        milestones = train_config.get('milestones', [30, 60, 90])
        gamma = train_config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    else:
        scheduler = None
    
    # Warmup scheduler
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=warmup_epochs,
        )
        if scheduler is not None:
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = warmup_scheduler
    
    return scheduler


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(yaml.dump(config, default_flow_style=False))
    
    # Set random seed
    seed = config.get('seed', 42)
    deterministic = config.get('deterministic', True)
    seed_everything(seed, deterministic)
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(config['paths']['output_dir'])
    log_dir = Path(config['paths']['log_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output dir
    shutil.copy(args.config, output_dir / 'config.yaml')
    
    # Setup data
    print("\n=== Setting up data ===")
    data_config = config.get('data', {})
    datamodule = CIFAR100DataModule(
        data_dir=config['paths'].get('data_dir', './data'),
        batch_size=config['train']['batch_size'],
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        val_split=0.1,
        use_randaugment=data_config.get('randaugment', True),
        randaugment_n=data_config.get('randaugment_n', 2),
        randaugment_m=data_config.get('randaugment_m', 10),
    )
    
    datamodule.prepare_data()
    datamodule.setup()
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    print(f"Train samples: {len(datamodule.train_dataset)}")
    print(f"Val samples: {len(datamodule.val_dataset) if datamodule.val_dataset else 0}")
    print(f"Batch size: {config['train']['batch_size']}")
    
    # Build model
    print("\n=== Building model ===")
    model = build_model(config)
    model = model.to(device)
    print(f"Model: {config['model']['name']}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    
    # Loss function
    label_smoothing = config['train'].get('label_smoothing', 0.0)
    if label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # AMP scaler
    use_amp = config['train'].get('amp', True) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"Mixed precision training: {use_amp}")
    
    # Setup logging
    writer = SummaryWriter(log_dir)
    metrics_logger = MetricsLogger(output_dir)
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc1 = 0.0
    
    if args.resume:
        print(f"\n=== Resuming from checkpoint: {args.resume} ===")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc1 = checkpoint.get('best_metric', 0.0)
    
    # Training loop
    print("\n=== Starting training ===")
    epochs = config['train']['epochs']
    
    for epoch in range(start_epoch, epochs):
        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            clip_grad=config['train'].get('gradient_clip', 0.0),
            mixup_alpha=config['train'].get('mixup_alpha', 0.0),
            cutmix_alpha=config['train'].get('cutmix_alpha', 0.0),
        )
        
        # Validate
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            desc='Val',
        )
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        metrics = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_metrics['loss'],
            'train_acc1': train_metrics['acc1'],
            'train_acc5': train_metrics['acc5'],
            'val_loss': val_metrics['loss'],
            'val_acc1': val_metrics['acc1'],
            'val_acc5': val_metrics['acc5'],
        }
        
        metrics_logger.log(metrics, epoch)
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Acc1/train', train_metrics['acc1'], epoch)
        writer.add_scalar('Acc1/val', val_metrics['acc1'], epoch)
        writer.add_scalar('Acc5/train', train_metrics['acc5'], epoch)
        writer.add_scalar('Acc5/val', val_metrics['acc5'], epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        # Print summary
        print_metrics_summary(train_metrics, val_metrics, epoch)
        
        # Save checkpoint
        is_best = val_metrics['acc1'] > best_acc1
        best_acc1 = max(val_metrics['acc1'], best_acc1)
        
        save_checkpoint(
            state={
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_metric': best_acc1,
                'config': config,
            },
            filepath=output_dir / 'last_model.pth',
            is_best=is_best,
        )
    
    # Final evaluation
    print("\n=== Final evaluation ===")
    test_loader = datamodule.test_dataloader()
    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        desc='Test',
    )
    print(f"Test Acc@1: {test_metrics['acc1']:.2f}%")
    print(f"Test Acc@5: {test_metrics['acc5']:.2f}%")
    
    # Plot metrics
    metrics_logger.plot_metrics(output_dir / 'training_curves.png')
    
    writer.close()
    print(f"\n=== Training complete! Best Val Acc@1: {best_acc1:.2f}% ===")


if __name__ == '__main__':
    main()
