"""
Evaluate a trained model and apply Temperature Scaling calibration.

Usage:
  python src/eval_and_calibrate.py --config configs/tinyimagenet_resnet50.yaml \
      --checkpoint runs/tinyimagenet_resnet50/last_model.pth --device cpu

This script:
 - Builds the datamodule (ImageFolder for Tiny-ImageNet)
 - Loads the model and checkpoint
 - Evaluates accuracy, NLL and ECE on the test set (pre-calibration)
 - Fits a TemperatureScaling on the validation set and saves temperature to output_dir
 - Evaluates the calibrated model on the test set (post-calibration)
"""
import argparse
import yaml
from pathlib import Path
import torch

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.common import get_device, load_checkpoint
from src.datasets.imagefolder import ImageFolderDataModule
from src.datasets.cifar100 import CIFAR100DataModule
from src.datasets.cifar10 import CIFAR10DataModule
from src.models.resnet import build_resnet_model
from src.models.vit import build_vit_model
from src.calibration.temperature_scaling import TemperatureScaling, evaluate_calibration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_datamodule(config):
    dataset = config.get('dataset', 'cifar100').lower()
    data_cfg = config.get('data', {})
    paths = config.get('paths', {})
    data_dir = paths.get('data_dir', './data')

    if dataset == 'imagefolder':
        dm = ImageFolderDataModule(
            data_dir=data_dir,
            img_size=data_cfg.get('img_size', config.get('model', {}).get('img_size', 224)),
            batch_size=config.get('eval', {}).get('batch_size', 256),
            num_workers=data_cfg.get('num_workers', 4),
            pin_memory=data_cfg.get('pin_memory', True),
            val_split=data_cfg.get('val_split', 0.1),
        )
    elif dataset == 'cifar100':
        dm = CIFAR100DataModule(
            data_dir=data_dir,
            batch_size=config.get('eval', {}).get('batch_size', 256),
            num_workers=data_cfg.get('num_workers', 4),
            pin_memory=data_cfg.get('pin_memory', True),
        )
    elif dataset == 'cifar10':
        dm = CIFAR10DataModule(
            data_dir=data_dir,
            batch_size=config.get('eval', {}).get('batch_size', 256),
            num_workers=data_cfg.get('num_workers', 4),
            pin_memory=data_cfg.get('pin_memory', True),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    dm.prepare_data()
    dm.setup()
    return dm


def build_model_from_config(config):
    model_cfg = config['model']
    name = model_cfg['name']
    # Build kwargs excluding the model name to avoid duplicate keys
    model_kwargs = {k: v for k, v in model_cfg.items() if k != 'name'}

    if name in ['resnet50', 'wide_resnet28_10']:
        return build_resnet_model(name, **model_kwargs)
    elif name in ['vit_tiny', 'deit_tiny']:
        return build_vit_model(name, **model_kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device(args.device)

    print(f"Using device: {device}")

    # Data
    print("Setting up data module...")
    dm = build_datamodule(config)

    if args.split == 'val':
        eval_loader = dm.val_dataloader()
    else:
        eval_loader = dm.test_dataloader()

    val_loader = dm.val_dataloader()

    # Model
    print("Building model...")
    model = build_model_from_config(config)
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, device=device)

    # Pre-calibration evaluation
    print("\n=== Pre-calibration evaluation ===")
    pre_metrics = evaluate_calibration(model, eval_loader, device=device)
    print(f"Pre-calibration: accuracy={pre_metrics['accuracy']:.4f}, ece={pre_metrics['ece']:.6f}, nll={pre_metrics['nll']:.6f}")

    # Fit temperature scaling on validation set
    print("\nFitting Temperature Scaling on validation set...")
    temp_scaler = TemperatureScaling()
    best_temp = temp_scaler.fit(model, val_loader, device=device)

    # Save temperature
    output_dir = Path(config['paths'].get('output_dir', './runs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = output_dir / 'temperature_scaler.pth'
    torch.save({'temperature': temp_scaler.temperature.detach().cpu()}, temp_path)
    print(f"Saved temperature scalar to {temp_path}")

    # Evaluate calibrated model
    print("\n=== Post-calibration evaluation ===")
    calibrated_model = temp_scaler.calibrate_model(model)
    calibrated_model = calibrated_model.to(device)
    post_metrics = evaluate_calibration(calibrated_model, eval_loader, device=device)
    print(f"Post-calibration: accuracy={post_metrics['accuracy']:.4f}, ece={post_metrics['ece']:.6f}, nll={post_metrics['nll']:.6f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
