"""
Calibration runner: evaluate a trained model, fit Temperature Scaling on validation set,
apply it, save calibrated model and reliability diagrams.

Usage (example):
  python src\calibrate.py --config configs\cifar100_wrn2810.yaml --checkpoint runs\cifar_wrn2810\best_model.pth
"""
import argparse
import yaml
import sys
from pathlib import Path

import torch

# Ensure project root is on path
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.cifar100 import CIFAR100DataModule
from src.models.resnet import build_resnet_model
from src.models.vit import build_vit_model
from src.utils.common import seed_everything, load_checkpoint, get_device
from src.calibration.temperature_scaling import TemperatureScaling, evaluate_calibration, plot_reliability_diagram


def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate a classification model with Temperature Scaling')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save calibrated model and plots')
    return parser.parse_args()


def build_model_from_config(config):
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
def get_probs_and_labels(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    return all_probs, all_labels


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed_everything(config.get('seed', 42), deterministic=False)

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Setup data
    data_cfg = config.get('data', {})
    eval_cfg = config.get('eval', {})

    datamodule = CIFAR100DataModule(
        data_dir=config['paths'].get('data_dir', './data'),
        batch_size=eval_cfg.get('batch_size', 256),
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=data_cfg.get('pin_memory', True),
        val_split=0.1,
        use_randaugment=False,
    )

    datamodule.prepare_data()
    datamodule.setup()

    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # Build and load model
    model = build_model_from_config(config)
    model = model.to(device)
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, device=device)

    # Evaluate before calibration
    print('\n============================================================')
    print('BEFORE CALIBRATION')
    print('============================================================')
    val_metrics = evaluate_calibration(model, val_loader, device=device)
    print('\nValidation Set:')
    print(f"  Accuracy: {val_metrics['accuracy']:.2f}%")
    print(f"  ECE:      {val_metrics['ece']:.4f}")
    print(f"  NLL:      {val_metrics['nll']:.4f}")

    test_metrics = evaluate_calibration(model, test_loader, device=device)
    print('\nTest Set:')
    print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  ECE:      {test_metrics['ece']:.4f}")
    print(f"  NLL:      {test_metrics['nll']:.4f}")

    # Fit temperature scaling on validation set
    print('\n============================================================')
    print('APPLYING TEMPERATURE SCALING')
    print('============================================================')
    temp_scaler = TemperatureScaling()
    optimized_T = temp_scaler.fit(model, val_loader, device=device)

    # Apply calibrated model
    calibrated_model = temp_scaler.calibrate_model(model)
    calibrated_model = calibrated_model.to(device)

    # Prepare save directory
    save_dir = Path(args.save_dir) if args.save_dir else Path(config['paths'].get('output_dir', './runs'))
    calib_dir = save_dir / 'calibration'
    calib_dir.mkdir(parents=True, exist_ok=True)

    # Save calibrated model
    save_path = calib_dir / 'calibrated_model.pth'
    torch.save({'model_state_dict': calibrated_model.state_dict(), 'temperature': temp_scaler.temperature.detach().cpu()}, save_path)
    print(f"\nCalibrated model saved to {save_path}")

    # Evaluate after calibration
    print('\n============================================================')
    print('AFTER TEMPERATURE SCALING')
    print('============================================================')
    val_metrics_after = evaluate_calibration(calibrated_model, val_loader, device=device)
    print('\nValidation Set:')
    print(f"  Accuracy: {val_metrics_after['accuracy']:.2f}%")
    print(f"  ECE:      {val_metrics_after['ece']:.4f} (was {val_metrics['ece']:.4f})")
    print(f"  NLL:      {val_metrics_after['nll']:.4f} (was {val_metrics['nll']:.4f})")

    test_metrics_after = evaluate_calibration(calibrated_model, test_loader, device=device)
    print('\nTest Set:')
    print(f"  Accuracy: {test_metrics_after['accuracy']:.2f}%")
    print(f"  ECE:      {test_metrics_after['ece']:.4f} (was {test_metrics['ece']:.4f})")
    print(f"  NLL:      {test_metrics_after['nll']:.4f} (was {test_metrics['nll']:.4f})")

    # Generate and save reliability diagrams (validation before/after)
    print('\nGenerating reliability diagrams...')
    # Get probs and labels before and after
    probs_val_before, labels_val = get_probs_and_labels(model, val_loader, device)
    probs_val_after, _ = get_probs_and_labels(calibrated_model, val_loader, device)

    plot_reliability_diagram(probs_val_before, labels_val, n_bins=15, save_path=str(calib_dir / 'reliability_val_before.png'))
    plot_reliability_diagram(probs_val_after, labels_val, n_bins=15, save_path=str(calib_dir / 'reliability_val_after.png'))

    probs_test_before, labels_test = get_probs_and_labels(model, test_loader, device)
    probs_test_after, _ = get_probs_and_labels(calibrated_model, test_loader, device)

    plot_reliability_diagram(probs_test_before, labels_test, n_bins=15, save_path=str(calib_dir / 'reliability_test_before.png'))
    plot_reliability_diagram(probs_test_after, labels_test, n_bins=15, save_path=str(calib_dir / 'reliability_test_after.png'))

    print(f"Reliability diagrams saved to {calib_dir}")


if __name__ == '__main__':
    main()
