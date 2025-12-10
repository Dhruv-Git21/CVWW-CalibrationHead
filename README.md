# Baseline Training – CIFAR-100 Classification + Object Detection

This repository provides reproducible baseline training scripts for:
1. **CIFAR-100 Classification**: ResNet-50, WideResNet-28-10, ViT-Tiny/DeiT-Tiny
2. **Object Detection**: RetinaNet on Pascal VOC 2007+2012 or mini-COCO

## Features
- **Modern Training**: Mixed precision (AMP), cosine LR scheduling, warmup, label smoothing
- **Data Augmentation**: RandAugment, CutMix, Mixup support
- **Reproducibility**: Fixed seeds, deterministic flags, config-driven experiments
- **Logging**: TensorBoard + CSV metrics, checkpoint management
- **Evaluation**: Top-1/Top-5 accuracy, confusion matrix for classification; mAP for detection

## Requirements

Python 3.10+ recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### CIFAR-100 Classification

**Train ResNet-50:**
```bash
python src/train_cls.py --config configs/cifar100_resnet50.yaml
```

**Train WideResNet-28-10:**
```bash
python src/train_cls.py --config configs/cifar100_wrn2810.yaml
```

**Train ViT-Tiny:**
```bash
python src/train_cls.py --config configs/cifar100_vit_tiny.yaml
```

**Evaluate a checkpoint:**
```bash
python src/eval_cls.py --config configs/cifar100_resnet50.yaml --checkpoint runs/cifar_resnet50/best_model.pth
```

### Object Detection

**Train RetinaNet on Pascal VOC:**
```bash
python src/train_det.py --config configs/detector_retinanet_voc.yaml
```

**Evaluate detector:**
```bash
python src/eval_det.py --config configs/detector_retinanet_voc.yaml --checkpoint runs/retinanet_voc/best_model.pth
```

## Repository Structure

```
.
├── configs/                      # YAML configuration files
│   ├── cifar100_resnet50.yaml
│   ├── cifar100_wrn2810.yaml
│   ├── cifar100_vit_tiny.yaml
│   └── detector_retinanet_voc.yaml
├── data/                         # Datasets (auto-downloaded)
├── src/
│   ├── datasets/                 # Dataset loaders
│   │   ├── cifar100.py
│   │   ├── voc.py
│   │   └── coco_subset.py
│   ├── models/                   # Model definitions
│   │   ├── resnet.py
│   │   ├── vit.py
│   │   └── retinanet.py
│   ├── utils/                    # Utilities
│   │   ├── common.py
│   │   ├── train_utils.py
│   │   ├── detector_utils.py
│   │   ├── metrics.py
│   │   └── transforms.py
│   ├── train_cls.py              # Train classification models
│   ├── eval_cls.py               # Evaluate classification models
│   ├── train_det.py              # Train detection models
│   └── eval_det.py               # Evaluate detection models
├── scripts/                      # Shell scripts for training
│   ├── run_cifar_resnet50.sh
│   ├── run_cifar_wrn2810.sh
│   ├── run_cifar_vit_tiny.sh
│   └── run_retinanet_voc.sh
├── requirements.txt
└── README.md
```

## Configuration

All experiments are config-driven via YAML files. Key parameters:

### Classification Config Example
```yaml
task: classification
dataset: cifar100
model:
  name: resnet50
  pretrained: false
  num_classes: 100
train:
  epochs: 200
  batch_size: 128
  optimizer: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 0.0005
  scheduler: cosine
  warmup_epochs: 5
  amp: true
  label_smoothing: 0.1
paths:
  output_dir: runs/cifar_resnet50
  log_dir: runs/cifar_resnet50/logs
seed: 42
```

## Models

### Classification
- **ResNet-50**: Standard torchvision ResNet-50 adapted for CIFAR-100
- **WideResNet-28-10**: Wide residual network with depth 28 and width factor 10
- **ViT-Tiny/DeiT-Tiny**: Vision transformer from timm library

### Detection
- **RetinaNet**: RetinaNet with ResNet-50 FPN backbone for object detection

## Training Details

### CIFAR-100
- **Transforms**: RandomCrop(32, padding=4), RandomHorizontalFlip, RandAugment, Normalize
- **Optimizer**: SGD (momentum=0.9, weight_decay=5e-4) for CNNs; AdamW for ViT
- **Schedule**: Cosine annealing with 5-epoch warmup
- **Epochs**: 200 (CNNs), 300 (ViT)
- **Batch Size**: 128 (CNNs), 256 (ViT)
- **Mixed Precision**: Enabled via torch.cuda.amp

### Detection
- **Dataset**: Pascal VOC 2007+2012 or mini-COCO subset
- **Optimizer**: SGD with momentum
- **Schedule**: Multi-step LR or cosine
- **Metrics**: COCO mAP via pycocotools

## Outputs

Each training run creates:
- `runs/{experiment_name}/`: Output directory
  - `best_model.pth`: Best checkpoint (highest val accuracy/mAP)
  - `last_model.pth`: Latest checkpoint
  - `logs/`: TensorBoard logs
  - `metrics.csv`: Training metrics in CSV format
  - `config.yaml`: Copy of the config used

## Reproducibility

- Fixed random seeds (Python, NumPy, PyTorch)
- Deterministic CUDA operations (may impact performance)
- All hyperparameters stored in config files
- Checkpoints include optimizer state, epoch, best metric

## Citation

If you use this codebase, please cite the original papers for the models and datasets used.

## License

MIT License
