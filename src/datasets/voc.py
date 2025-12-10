"""
Pascal VOC dataset loader for object detection.
"""
import os
import xml.etree.ElementTree as ET
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import numpy as np


class VOCDetection(Dataset):
    """Pascal VOC dataset for object detection."""
    
    # VOC class names (20 classes + background)
    CLASSES = [
        '__background__',
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(
        self,
        root: str,
        year: str = '2007',
        image_set: str = 'train',
        transform: Optional[callable] = None,
    ):
        """
        Args:
            root: Root directory of VOC dataset (VOCdevkit/)
            year: Dataset year ('2007', '2012', or '2007+2012')
            image_set: 'train', 'val', 'trainval', or 'test'
            transform: Optional transform to apply
        """
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        
        # Handle combined dataset
        if year == '2007+2012':
            self.years = ['2007', '2012']
        else:
            self.years = [year]
        
        # Load image IDs
        self.images = []
        self.annotations = []
        
        for yr in self.years:
            voc_root = os.path.join(root, f'VOC{yr}')
            image_set_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{image_set}.txt')
            
            if not os.path.exists(image_set_file):
                raise FileNotFoundError(f"Image set file not found: {image_set_file}")
            
            with open(image_set_file, 'r') as f:
                file_names = [x.strip() for x in f.readlines()]
            
            for file_name in file_names:
                img_path = os.path.join(voc_root, 'JPEGImages', f'{file_name}.jpg')
                anno_path = os.path.join(voc_root, 'Annotations', f'{file_name}.xml')
                
                if os.path.exists(img_path) and os.path.exists(anno_path):
                    self.images.append(img_path)
                    self.annotations.append(anno_path)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            image: Tensor of shape (C, H, W)
            target: Dict with keys 'boxes', 'labels', 'image_id', 'area', 'iscrowd'
        """
        # Load image
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Parse annotation
        anno_path = self.annotations[idx]
        boxes, labels, difficulties = self._parse_annotation(anno_path)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Calculate area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Apply transforms
        if self.transform is not None:
            img, target = self.transform(img, target)
        else:
            img = T.ToTensor()(img)
        
        return img, target
    
    def _parse_annotation(self, xml_path: str) -> Tuple[List, List, List]:
        """Parse VOC XML annotation."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        difficulties = []
        
        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
            class_name = obj.find('name').text.strip()
            
            if class_name not in self.class_to_idx:
                continue
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[class_name])
            difficulties.append(difficult)
        
        return boxes, labels, difficulties


class VOCDataModule:
    """DataModule for VOC detection."""
    
    def __init__(
        self,
        data_dir: str,
        year: str = '2007',
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self.data_dir = data_dir
        self.year = year
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.num_classes = 21  # 20 + background
    
    def setup(self):
        """Setup train/val datasets."""
        self.train_dataset = VOCDetection(
            root=self.data_dir,
            year=self.year,
            image_set='trainval' if self.year != '2007+2012' else 'train',
            transform=self.get_transform(train=True)
        )
        
        self.val_dataset = VOCDetection(
            root=self.data_dir,
            year='2007',  # Always validate on VOC2007 test
            image_set='test',
            transform=self.get_transform(train=False)
        )
    
    def get_transform(self, train: bool = True):
        """Get transforms for detection."""
        # For detection, we typically just normalize
        # Augmentation is often done inside the model or with specialized libs
        def transform(image, target):
            image = T.ToTensor()(image)
            return image, target
        
        return transform
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for detection."""
        return tuple(zip(*batch))
