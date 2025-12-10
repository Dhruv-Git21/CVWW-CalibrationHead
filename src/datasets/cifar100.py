"""
CIFAR-100 dataset loader with transforms and utilities.
"""
import os
from typing import Tuple, Optional, Callable
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment


class CIFAR100DataModule:
    """DataModule-like loader for CIFAR-100."""
    
    # CIFAR-100 statistics
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.1,
        use_randaugment: bool = True,
        randaugment_n: int = 2,
        randaugment_m: int = 10,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.use_randaugment = use_randaugment
        self.randaugment_n = randaugment_n
        self.randaugment_m = randaugment_m
        
        self.num_classes = 100
        
    def get_transforms(self, train: bool = True) -> transforms.Compose:
        """Get transforms for train/val/test."""
        if train:
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            if self.use_randaugment:
                transform_list.append(RandAugment(num_ops=self.randaugment_n, magnitude=self.randaugment_m))
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])
            return transforms.Compose(transform_list)
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])
    
    def prepare_data(self):
        """Download CIFAR-100 if not present."""
        datasets.CIFAR100(root=self.data_dir, train=True, download=True)
        datasets.CIFAR100(root=self.data_dir, train=False, download=True)
    
    def setup(self):
        """Setup train/val/test datasets."""
        # Full training set with augmentation
        train_dataset_full = datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=False,
            transform=self.get_transforms(train=True)
        )
        
        # Split into train and validation
        if self.val_split > 0:
            val_size = int(len(train_dataset_full) * self.val_split)
            train_size = len(train_dataset_full) - val_size
            
            # Use generator for reproducibility
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset_temp = random_split(
                train_dataset_full, [train_size, val_size], generator=generator
            )
            
            # Create validation dataset with eval transforms
            val_dataset_base = datasets.CIFAR100(
                root=self.data_dir,
                train=True,
                download=False,
                transform=self.get_transforms(train=False)
            )
            
            # Create subset with validation indices but eval transforms
            val_dataset = torch.utils.data.Subset(val_dataset_base, val_dataset_temp.indices)
            
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        else:
            self.train_dataset = train_dataset_full
            self.val_dataset = None
        
        # Test dataset
        self.test_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=False,
            transform=self.get_transforms(train=False)
        )
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    @staticmethod
    def get_class_names():
        """Get CIFAR-100 class names."""
        # These are the coarse and fine labels
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
            'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]
