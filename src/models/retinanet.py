"""
RetinaNet model for object detection.
"""
import torch
import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.anchor_utils import AnchorGenerator


def get_retinanet_resnet50_fpn(
    num_classes: int = 91,
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    min_size: int = 600,
    max_size: int = 1000,
    **kwargs
):
    """
    Get RetinaNet with ResNet-50 FPN backbone.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained model
        pretrained_backbone: Whether to use pretrained backbone
        min_size: Minimum size of the image to be rescaled
        max_size: Maximum size of the image to be rescaled
        **kwargs: Additional arguments
    """
    if pretrained:
        # Load pretrained RetinaNet
        model = retinanet_resnet50_fpn_v2(pretrained=True)
        
        # Replace classification head if num_classes differs
        if num_classes != 91:  # COCO has 91 classes
            in_features = model.head.classification_head.conv[0].in_channels
            num_anchors = model.head.classification_head.num_anchors
            
            model.head.classification_head = RetinaNetClassificationHead(
                in_channels=in_features,
                num_anchors=num_anchors,
                num_classes=num_classes,
            )
    else:
        # Create model from scratch
        model = retinanet_resnet50_fpn_v2(
            pretrained=False,
            pretrained_backbone=pretrained_backbone,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )
    
    return model


def get_retinanet_model(
    model_name: str = 'retinanet_resnet50_fpn',
    num_classes: int = 91,
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    **kwargs
):
    """
    Build RetinaNet model by name.
    
    Args:
        model_name: Model architecture name
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained model
        pretrained_backbone: Whether to use pretrained backbone
        **kwargs: Additional model-specific arguments
    """
    min_size = kwargs.get('min_size', 600)
    max_size = kwargs.get('max_size', 1000)
    
    if model_name == 'retinanet_resnet50_fpn':
        return get_retinanet_resnet50_fpn(
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            min_size=min_size,
            max_size=max_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


class RetinaNetWrapper(nn.Module):
    """
    Wrapper for RetinaNet to handle training and inference modes.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, images, targets=None):
        """
        Args:
            images: List of images or batched tensor
            targets: List of target dicts (only for training)
        
        Returns:
            During training: dict of losses
            During inference: list of detections
        """
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            return self.model(images, targets)
        else:
            return self.model(images)


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation
    print("Creating RetinaNet for VOC...")
    model = get_retinanet_resnet50_fpn(
        num_classes=21,  # VOC: 20 classes + background
        pretrained=False,
        pretrained_backbone=True
    )
    print(f"Number of parameters: {count_parameters(model):,}")
    
    # Test forward pass (inference mode)
    model.eval()
    images = [torch.randn(3, 600, 800), torch.randn(3, 500, 700)]
    with torch.no_grad():
        outputs = model(images)
    print(f"Number of images: {len(images)}")
    print(f"Number of outputs: {len(outputs)}")
    print(f"Output keys: {outputs[0].keys()}")
