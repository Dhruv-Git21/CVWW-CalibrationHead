"""
Utility to create a mini-COCO subset for faster experimentation.
"""
import os
import json
import shutil
from typing import List, Optional
from pathlib import Path
import random


class COCOMiniSubset:
    """Create a mini COCO subset with a limited number of images."""
    
    def __init__(
        self,
        coco_root: str,
        output_root: str,
        num_images: int = 1000,
        seed: int = 42
    ):
        """
        Args:
            coco_root: Root directory of full COCO dataset
            output_root: Directory to save mini subset
            num_images: Number of images to include
            seed: Random seed
        """
        self.coco_root = Path(coco_root)
        self.output_root = Path(output_root)
        self.num_images = num_images
        self.seed = seed
        
        random.seed(seed)
    
    def create_subset(self, split: str = 'train2017'):
        """Create mini subset for given split."""
        print(f"Creating mini COCO subset with {self.num_images} images from {split}")
        
        # Load original annotations
        anno_file = self.coco_root / 'annotations' / f'instances_{split}.json'
        with open(anno_file, 'r') as f:
            coco_data = json.load(f)
        
        # Randomly sample images
        all_images = coco_data['images']
        selected_images = random.sample(all_images, min(self.num_images, len(all_images)))
        selected_image_ids = {img['id'] for img in selected_images}
        
        # Filter annotations
        selected_annotations = [
            ann for ann in coco_data['annotations']
            if ann['image_id'] in selected_image_ids
        ]
        
        # Create output directories
        output_images_dir = self.output_root / split
        output_images_dir.mkdir(parents=True, exist_ok=True)
        
        output_anno_dir = self.output_root / 'annotations'
        output_anno_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        print("Copying images...")
        source_images_dir = self.coco_root / split
        for img in selected_images:
            src = source_images_dir / img['file_name']
            dst = output_images_dir / img['file_name']
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        
        # Save filtered annotations
        mini_coco_data = {
            'info': coco_data['info'],
            'licenses': coco_data['licenses'],
            'categories': coco_data['categories'],
            'images': selected_images,
            'annotations': selected_annotations
        }
        
        output_anno_file = output_anno_dir / f'instances_{split}_mini.json'
        with open(output_anno_file, 'w') as f:
            json.dump(mini_coco_data, f)
        
        print(f"Mini subset created:")
        print(f"  - Images: {len(selected_images)}")
        print(f"  - Annotations: {len(selected_annotations)}")
        print(f"  - Saved to: {self.output_root}")
    
    def create_train_val_subsets(
        self,
        train_images: int = 800,
        val_images: int = 200
    ):
        """Create both train and val mini subsets."""
        # Save original num_images
        original_num = self.num_images
        
        # Create train subset
        self.num_images = train_images
        self.create_subset('train2017')
        
        # Create val subset
        self.num_images = val_images
        self.create_subset('val2017')
        
        # Restore
        self.num_images = original_num
        
        print(f"\nMini COCO dataset created successfully!")


def create_mini_coco(
    coco_root: str = './data/coco',
    output_root: str = './data/coco_mini',
    num_train: int = 800,
    num_val: int = 200,
    seed: int = 42
):
    """
    Convenience function to create mini COCO subset.
    
    Usage:
        from src.datasets.coco_subset import create_mini_coco
        create_mini_coco(
            coco_root='./data/coco',
            output_root='./data/coco_mini',
            num_train=800,
            num_val=200
        )
    """
    subset_creator = COCOMiniSubset(
        coco_root=coco_root,
        output_root=output_root,
        num_images=num_train,
        seed=seed
    )
    subset_creator.create_train_val_subsets(train_images=num_train, val_images=num_val)


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Create mini COCO subset')
    parser.add_argument('--coco-root', type=str, default='./data/coco',
                        help='Root directory of full COCO dataset')
    parser.add_argument('--output-root', type=str, default='./data/coco_mini',
                        help='Output directory for mini subset')
    parser.add_argument('--num-train', type=int, default=800,
                        help='Number of training images')
    parser.add_argument('--num-val', type=int, default=200,
                        help='Number of validation images')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    create_mini_coco(
        coco_root=args.coco_root,
        output_root=args.output_root,
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed
    )
