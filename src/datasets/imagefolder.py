"""
Generic ImageFolder DataModule for datasets arranged in train/val/test folders.

This supports Tiny-ImageNet and other ImageNet-style datasets.
"""
import os
from typing import Optional
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class ImageFolderDataModule:
    """DataModule using torchvision.datasets.ImageFolder.

    Expects directory structure like:
      data_dir/train/<class>/*.jpg
      data_dir/val/<class>/*.jpg
      data_dir/test/<class>/*.jpg

    If explicit val/test folders are not present, a val split is created from train.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        data_dir: str = './data',
        train_dir: str = 'train',
        val_dir: str = 'val',
        test_dir: str = 'test',
        img_size: int = 224,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.1,
        mean: tuple = None,
        std: tuple = None,
    ):
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.mean = mean or self.IMAGENET_MEAN
        self.std = std or self.IMAGENET_STD

        self.num_classes = None

    def get_transforms(self, train: bool = True):
        if train:
            return transforms.Compose([
                transforms.Resize(int(self.img_size * 1.15)),
                transforms.RandomResizedCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(int(self.img_size * 1.15)),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

    def prepare_data(self):
        # No automatic download for ImageFolder datasets
        return

    def setup(self):
        """Create datasets for train/val/test."""
        train_path = os.path.join(self.data_dir, self.train_dir)
        val_path = os.path.join(self.data_dir, self.val_dir)
        test_path = os.path.join(self.data_dir, self.test_dir)

        if os.path.isdir(train_path):
            train_dataset_full = datasets.ImageFolder(train_path, transform=self.get_transforms(train=True))
        else:
            raise FileNotFoundError(f"Train directory not found: {train_path}")

        # If val folder exists, use it; else split from train
        if os.path.isdir(val_path):
            val_dataset = datasets.ImageFolder(val_path, transform=self.get_transforms(train=False))
            train_dataset = train_dataset_full
        else:
            if self.val_split > 0:
                val_size = int(len(train_dataset_full) * self.val_split)
                train_size = len(train_dataset_full) - val_size
                generator = torch.Generator().manual_seed(42)
                train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size], generator=generator)
                # Ensure val uses evaluation transforms
                val_dataset.dataset.transform = self.get_transforms(train=False)
            else:
                train_dataset = train_dataset_full
                val_dataset = None

        # Test dataset if present
        if os.path.isdir(test_path):
            test_dataset = datasets.ImageFolder(test_path, transform=self.get_transforms(train=False))
        else:
            test_dataset = None

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Set num_classes
        if hasattr(train_dataset_full, 'classes'):
            self.num_classes = len(train_dataset_full.classes)
        else:
            # try to infer from train_dataset (Subset)
            try:
                self.num_classes = len(train_dataset.dataset.classes)
            except Exception:
                self.num_classes = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers>0)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size*2, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers>0)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(self.test_dataset, batch_size=self.batch_size*2, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers>0)
