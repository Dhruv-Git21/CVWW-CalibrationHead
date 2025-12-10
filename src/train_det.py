"""
Training script for object detection models (RetinaNet on VOC/COCO).

Usage:
    python src/train_det.py --config configs/detector_retinanet_voc.yaml
"""
import argparse
import yaml
import os
import sys
from pathlib import Path
import shutil

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.voc import VOCDataModule
from src.models.retinanet import get_retinanet_model
from src.utils.common import seed_everything, save_checkpoint, load_checkpoint, get_device, count_parameters
from src.utils.detector_utils import train_one_epoch_detection, evaluate_detection
from src.utils.metrics import MetricsLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Train object detection model')
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
    
    model = get_retinanet_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=model_config.get('pretrained', False),
        pretrained_backbone=model_config.get('pretrained_backbone', True),
        min_size=model_config.get('min_size', 600),
        max_size=model_config.get('max_size', 1000),
    )
    
    return model


def build_optimizer(model, config):
    """Build optimizer based on config."""
    train_config = config['train']
    optimizer_name = train_config['optimizer'].lower()
    lr = train_config['lr']
    weight_decay = train_config['weight_decay']
    
    # Get parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=train_config.get('momentum', 0.9),
            weight_decay=weight_decay,
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def build_scheduler(optimizer, config):
    """Build learning rate scheduler."""
    train_config = config['train']
    scheduler_name = train_config.get('scheduler', 'multi_step')
    
    if scheduler_name == 'multi_step':
        milestones = train_config.get('milestones', [30, 40])
        gamma = train_config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    elif scheduler_name == 'cosine':
        epochs = train_config['epochs']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=0,
        )
    else:
        scheduler = None
    
    # Warmup
    warmup_epochs = train_config.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        warmup_factor = train_config.get('warmup_factor', 0.001)
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_factor,
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
    deterministic = config.get('deterministic', False)
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
    
    if config['dataset'] == 'voc':
        datamodule = VOCDataModule(
            data_dir=config['paths'].get('data_dir', './data/VOCdevkit'),
            year=data_config.get('voc_year', '2007'),
            batch_size=config['train']['batch_size'],
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
        )
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    print(f"Train samples: {len(datamodule.train_dataset)}")
    print(f"Val samples: {len(datamodule.val_dataset)}")
    print(f"Batch size: {config['train']['batch_size']}")
    
    # Build model
    print("\n=== Building model ===")
    model = build_model(config)
    model = model.to(device)
    print(f"Model: {config['model']['name']}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    
    # AMP scaler
    use_amp = config['train'].get('amp', True) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"Mixed precision training: {use_amp}")
    
    # Setup logging
    writer = SummaryWriter(log_dir)
    metrics_logger = MetricsLogger(output_dir)
    
    # Resume from checkpoint
    start_epoch = 0
    best_map = 0.0
    
    if args.resume:
        print(f"\n=== Resuming from checkpoint: {args.resume} ===")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_map = checkpoint.get('best_metric', 0.0)
    
    # Training loop
    print("\n=== Starting training ===")
    epochs = config['train']['epochs']
    
    for epoch in range(start_epoch, epochs):
        # Train
        train_metrics = train_one_epoch_detection(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            clip_grad=config['train'].get('gradient_clip', 0.0),
        )
        
        # Validate (every N epochs to save time)
        eval_every = 5
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            val_metrics = evaluate_detection(
                model=model,
                dataloader=val_loader,
                device=device,
                iou_threshold=config['eval'].get('iou_threshold', 0.5),
            )
        else:
            val_metrics = {'mAP@0.5': 0.0}
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        metrics = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_metrics['loss'],
        }
        if 'mAP@0.5' in val_metrics and val_metrics['mAP@0.5'] > 0:
            metrics['val_mAP@0.5'] = val_metrics['mAP@0.5']
        
        metrics_logger.log(metrics, epoch)
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        if 'mAP@0.5' in val_metrics and val_metrics['mAP@0.5'] > 0:
            writer.add_scalar('mAP/val', val_metrics['mAP@0.5'], epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        if 'mAP@0.5' in val_metrics and val_metrics['mAP@0.5'] > 0:
            print(f"  Val mAP@0.5: {val_metrics['mAP@0.5']:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save checkpoint
        current_map = val_metrics.get('mAP@0.5', 0.0)
        is_best = current_map > best_map and current_map > 0
        best_map = max(current_map, best_map)
        
        save_checkpoint(
            state={
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_metric': best_map,
                'config': config,
            },
            filepath=output_dir / 'last_model.pth',
            is_best=is_best,
        )
    
    # Final evaluation
    print("\n=== Final evaluation ===")
    final_metrics = evaluate_detection(
        model=model,
        dataloader=val_loader,
        device=device,
        iou_threshold=config['eval'].get('iou_threshold', 0.5),
    )
    print(f"Final mAP@0.5: {final_metrics.get('mAP@0.5', 0.0):.4f}")
    
    writer.close()
    print(f"\n=== Training complete! Best mAP@0.5: {best_map:.4f} ===")


if __name__ == '__main__':
    main()
