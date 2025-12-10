"""
ResNet and WideResNet models for CIFAR-100 classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_resnet50(num_classes: int = 100, pretrained: bool = False):
    """
    Get ResNet-50 model adapted for CIFAR-100.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
    """
    model = models.resnet50(pretrained=pretrained)
    
    # Adapt for CIFAR-100 (32x32 images)
    # Replace first conv layer: 7x7 stride 2 -> 3x3 stride 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove max pooling (too aggressive for 32x32)
    model.maxpool = nn.Identity()
    
    # Replace final fc layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


class BasicBlock(nn.Module):
    """Basic block for WideResNet."""
    
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout_rate = dropout_rate
        self.equal_in_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.equal_in_out) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride,
            padding=0, bias=False) or None
    
    def forward(self, x):
        if not self.equal_in_out:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        
        out = self.relu2(self.bn2(self.conv1(out if self.equal_in_out else x)))
        
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        
        out = self.conv2(out)
        
        return torch.add(x if self.equal_in_out else self.conv_shortcut(x), out)


class NetworkBlock(nn.Module):
    """Network block for WideResNet."""
    
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_rate)
    
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropout_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    WideResNet for CIFAR-100.
    
    Paper: "Wide Residual Networks" (https://arxiv.org/abs/1605.07146)
    """
    
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.0, num_classes=100):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'Depth should be 6n+4'
        n = (depth - 4) / 6
        block = BasicBlock
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        # 1st block
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1, dropout_rate)
        
        # 2nd block
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, dropout_rate)
        
        # 3rd block
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, dropout_rate)
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.n_channels = n_channels[3]
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.n_channels)
        return self.fc(out)


def get_wide_resnet28_10(num_classes: int = 100, dropout_rate: float = 0.0):
    """
    Get WideResNet-28-10 model for CIFAR-100.
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate (default: 0.0)
    """
    return WideResNet(depth=28, widen_factor=10, dropout_rate=dropout_rate, num_classes=num_classes)


def build_resnet_model(model_name: str, num_classes: int = 100, **kwargs):
    """
    Build ResNet/WideResNet model by name.
    
    Args:
        model_name: 'resnet50' or 'wide_resnet28_10'
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
    """
    if model_name == 'resnet50':
        return get_resnet50(num_classes=num_classes, pretrained=kwargs.get('pretrained', False))
    elif model_name == 'wide_resnet28_10':
        return get_wide_resnet28_10(num_classes=num_classes, dropout_rate=kwargs.get('dropout_rate', 0.0))
    else:
        raise ValueError(f"Unknown model: {model_name}")
