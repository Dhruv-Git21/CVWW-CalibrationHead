"""
Transform utilities for data augmentation.
"""
import torch
import torchvision.transforms as T
from torchvision.transforms import RandAugment, AutoAugment, AutoAugmentPolicy
from typing import Tuple, Optional


class CIFAR100Transforms:
    """Transform utilities for CIFAR-100."""
    
    # CIFAR-100 statistics
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)
    
    @staticmethod
    def get_train_transform(
        use_randaugment: bool = True,
        randaugment_n: int = 2,
        randaugment_m: int = 10,
        use_autoaugment: bool = False,
    ) -> T.Compose:
        """
        Get training transforms for CIFAR-100.
        
        Args:
            use_randaugment: Whether to use RandAugment
            randaugment_n: Number of augmentation operations
            randaugment_m: Magnitude of augmentation
            use_autoaugment: Whether to use AutoAugment
        
        Returns:
            Composed transforms
        """
        transforms = [
            T.RandomCrop(32, padding=4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
        ]
        
        if use_autoaugment:
            transforms.append(AutoAugment(AutoAugmentPolicy.CIFAR10))
        elif use_randaugment:
            transforms.append(RandAugment(num_ops=randaugment_n, magnitude=randaugment_m))
        
        transforms.extend([
            T.ToTensor(),
            T.Normalize(CIFAR100Transforms.MEAN, CIFAR100Transforms.STD),
        ])
        
        return T.Compose(transforms)
    
    @staticmethod
    def get_val_transform() -> T.Compose:
        """Get validation/test transforms for CIFAR-100."""
        return T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR100Transforms.MEAN, CIFAR100Transforms.STD),
        ])
    
    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor for visualization.
        
        Args:
            tensor: Normalized tensor (C, H, W)
        
        Returns:
            Denormalized tensor
        """
        mean = torch.tensor(CIFAR100Transforms.MEAN).view(3, 1, 1)
        std = torch.tensor(CIFAR100Transforms.STD).view(3, 1, 1)
        return tensor * std + mean


class DetectionTransforms:
    """Transform utilities for object detection."""
    
    @staticmethod
    def get_train_transform():
        """Get training transforms for detection."""
        # For detection, augmentation is typically handled by the model or specialized libs
        # We just normalize here
        return T.Compose([
            T.ToTensor(),
        ])
    
    @staticmethod
    def get_val_transform():
        """Get validation transforms for detection."""
        return T.Compose([
            T.ToTensor(),
        ])


class MixupCutmix:
    """
    Mixup and CutMix implementation.
    
    Can be used as a collate function or applied in training loop.
    """
    
    def __init__(
        self,
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        num_classes: int = 100,
    ):
        """
        Args:
            mixup_alpha: Mixup alpha parameter
            cutmix_alpha: CutMix alpha parameter
            prob: Probability of applying mixup/cutmix
            switch_prob: Probability of using mixup vs cutmix
            num_classes: Number of classes
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
    
    def __call__(self, batch):
        """
        Apply Mixup/CutMix to a batch.
        
        Args:
            batch: Batch of (image, label) tuples
        
        Returns:
            Mixed batch with soft labels
        """
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        # Skip if probability check fails
        if torch.rand(1).item() > self.prob:
            return images, labels
        
        # Choose between mixup and cutmix
        use_mixup = torch.rand(1).item() < self.switch_prob
        
        if use_mixup and self.mixup_alpha > 0:
            return self._mixup(images, labels)
        elif self.cutmix_alpha > 0:
            return self._cutmix(images, labels)
        else:
            return images, labels
    
    def _mixup(self, images, labels):
        """Apply Mixup."""
        import numpy as np
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Create soft labels
        labels_a = torch.nn.functional.one_hot(labels, self.num_classes).float()
        labels_b = torch.nn.functional.one_hot(labels[index], self.num_classes).float()
        mixed_labels = lam * labels_a + (1 - lam) * labels_b
        
        return mixed_images, mixed_labels
    
    def _cutmix(self, images, labels):
        """Apply CutMix."""
        import numpy as np
        
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # Get random box
        _, _, H, W = images.size()
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # Create soft labels
        labels_a = torch.nn.functional.one_hot(labels, self.num_classes).float()
        labels_b = torch.nn.functional.one_hot(labels[index], self.num_classes).float()
        mixed_labels = lam * labels_a + (1 - lam) * labels_b
        
        return mixed_images, mixed_labels
