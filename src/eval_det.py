"""
Evaluation script for object detection models.

Usage:
    python src/eval_det.py --config configs/detector_retinanet_voc.yaml --checkpoint runs/retinanet_voc/best_model.pth
"""
import argparse
import yaml
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.voc import VOCDataModule
from src.models.retinanet import get_retinanet_model
from src.utils.common import seed_everything, load_checkpoint, get_device
from src.utils.detector_utils import evaluate_detection


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate object detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--iou-threshold', type=float, default=None, help='IoU threshold for mAP')
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
        pretrained=False,
        pretrained_backbone=model_config.get('pretrained_backbone', True),
        min_size=model_config.get('min_size', 600),
        max_size=model_config.get('max_size', 1000),
    )
    
    return model


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Set random seed
    seed = config.get('seed', 42)
    seed_everything(seed, deterministic=False)
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Setup data
    print("\n=== Setting up data ===")
    data_config = config.get('data', {})
    eval_config = config.get('eval', {})
    
    if config['dataset'] == 'voc':
        datamodule = VOCDataModule(
            data_dir=config['paths'].get('data_dir', './data/VOCdevkit'),
            year=data_config.get('voc_year', '2007'),
            batch_size=eval_config.get('batch_size', 8),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
        )
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    datamodule.setup()
    val_loader = datamodule.val_dataloader()
    
    print(f"Val samples: {len(datamodule.val_dataset)}")
    print(f"Batch size: {eval_config.get('batch_size', 8)}")
    
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
    iou_threshold = args.iou_threshold or eval_config.get('iou_threshold', 0.5)
    
    metrics = evaluate_detection(
        model=model,
        dataloader=val_loader,
        device=device,
        iou_threshold=iou_threshold,
    )
    
    print(f"\n=== Results ===")
    print(f"mAP@{iou_threshold}: {metrics.get('mAP@0.5', 0.0):.4f}")
    if 'mAP@0.75' in metrics:
        print(f"mAP@0.75: {metrics['mAP@0.75']:.4f}")
    if 'mAP@0.5:0.95' in metrics:
        print(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    
    print(f"\nNumber of images: {metrics.get('num_images', 0)}")
    print(f"Total predictions: {metrics.get('num_predictions', 0)}")
    print(f"Total ground truth boxes: {metrics.get('num_targets', 0)}")
    
    print("\nNote: This is a simplified mAP calculation.")
    print("For full COCO-style evaluation, use pycocotools with COCO dataset.")
    
    print("\n=== Evaluation complete ===")


if __name__ == '__main__':
    main()
