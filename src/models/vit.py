"""
Vision Transformer (ViT) models for CIFAR-100 classification.
"""
import torch
import torch.nn as nn
import timm


def get_vit_tiny(
    num_classes: int = 100,
    img_size: int = 32,
    patch_size: int = 4,
    pretrained: bool = False,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
):
    """
    Get ViT-Tiny model adapted for CIFAR-100.
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        patch_size: Patch size for ViT
        pretrained: Whether to use pretrained weights
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
    """
    # Create ViT-Tiny model using timm
    # vit_tiny_patch16_224 is the base, but we adapt it for CIFAR
    model = timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
    )
    
    return model


def get_deit_tiny(
    num_classes: int = 100,
    img_size: int = 32,
    patch_size: int = 4,
    pretrained: bool = False,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
):
    """
    Get DeiT-Tiny model adapted for CIFAR-100.
    
    DeiT (Data-efficient image Transformers) includes distillation token.
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        patch_size: Patch size for ViT
        pretrained: Whether to use pretrained weights
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
    """
    # Create DeiT-Tiny model using timm
    model = timm.create_model(
        'deit_tiny_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
    )
    
    return model


def build_vit_model(model_name: str, num_classes: int = 100, **kwargs):
    """
    Build Vision Transformer model by name.
    
    Args:
        model_name: 'vit_tiny' or 'deit_tiny'
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
    """
    img_size = kwargs.get('img_size', 32)
    patch_size = kwargs.get('patch_size', 4)
    pretrained = kwargs.get('pretrained', False)
    drop_rate = kwargs.get('drop_rate', 0.0)
    attn_drop_rate = kwargs.get('attn_drop_rate', 0.0)
    drop_path_rate = kwargs.get('drop_path_rate', 0.1)
    
    if model_name == 'vit_tiny':
        return get_vit_tiny(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            pretrained=pretrained,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
    elif model_name == 'deit_tiny':
        return get_deit_tiny(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            pretrained=pretrained,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation
    print("Creating ViT-Tiny for CIFAR-100...")
    model = get_vit_tiny(num_classes=100, img_size=32, patch_size=4)
    print(f"Number of parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
