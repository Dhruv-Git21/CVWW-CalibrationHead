"""
Calibration Attention: Instance-wise Temperature Scaling for Vision Transformers

Reference: "Calibration Attention: Instance-wise Temperature Scaling 
           for Vision Transformers" (Liang et al., 2025)

Key Innovation: Instead of using a single global temperature T for all samples,
this method learns a per-instance temperature T_n = f(x_n) that adapts to each input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CalibrationAttention(nn.Module):
    """
    Calibration Attention module for Vision Transformers.
    
    This module adds instance-wise temperature scaling to the attention mechanism:
        Attention(Q, K, V, T) = softmax(QK^T / (sqrt(d_k) * T)) V
    
    Where T = f(CLS_token) is learned per-instance.
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 1/sqrt(d_k)
        
        # Standard attention components
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Temperature prediction network
        self.temp_predictor = TemperaturePredictor(dim)
    
    def forward(self, x):
        """
        Forward pass with calibrated attention.
        
        Args:
            x: Input tensor, shape (B, N, C)
               B = batch size
               N = number of tokens (including CLS token)
               C = embedding dimension
        
        Returns:
            Output tensor, shape (B, N, C)
        """
        B, N, C = x.shape
        
        # Predict instance-wise temperature from CLS token
        # CLS token is typically the first token (index 0)
        cls_token = x[:, 0]  # Shape: (B, C)
        temperature = self.temp_predictor(cls_token)  # Shape: (B, 1)
        
        # Standard multi-head attention computation
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # Compute attention scores with calibration
        # Standard: attn = (q @ k^T) * scale
        # Calibrated: attn = (q @ k^T) * scale / temperature
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply instance-wise temperature scaling
        # Reshape temperature for broadcasting: (B, 1) -> (B, 1, 1, 1)
        temperature = temperature.view(B, 1, 1, 1)
        attn = attn / temperature  # Divide by temperature
        
        # Softmax and dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TemperaturePredictor(nn.Module):
    """
    Network that predicts instance-wise temperature from CLS token.
    
    Architecture: CLS_token -> MLP -> Sigmoid -> Temperature
    
    The temperature is constrained to a reasonable range (e.g., 0.5 to 3.0)
    to ensure stable training.
    """
    
    def __init__(self, dim, hidden_dim=None, temp_min=0.5, temp_max=3.0):
        """
        Args:
            dim: Input dimension (CLS token dimension)
            hidden_dim: Hidden layer dimension (default: dim // 2)
            temp_min: Minimum temperature value
            temp_max: Maximum temperature value
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim // 2
        
        self.temp_min = temp_min
        self.temp_max = temp_max
        
        # MLP: Linear -> ReLU -> Linear -> Sigmoid
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, cls_token):
        """
        Predict temperature from CLS token.
        
        Args:
            cls_token: CLS token features, shape (B, C)
        
        Returns:
            temperature: Per-instance temperature, shape (B, 1)
                        Values in range [temp_min, temp_max]
        """
        # MLP output in [0, 1]
        t_normalized = self.mlp(cls_token)
        
        # Scale to [temp_min, temp_max]
        temperature = self.temp_min + (self.temp_max - self.temp_min) * t_normalized
        
        return temperature


class OutputTemperatureScaling(nn.Module):
    """
    Simpler variant: Apply instance-wise temperature scaling to final logits.
    
    Instead of modifying attention, this applies temperature to the output logits:
        p_i = exp(z_i/T_n) / sum_j exp(z_j/T_n)
    
    Where T_n = f(features_n) is predicted from the model's features.
    """
    
    def __init__(self, feature_dim, num_classes, temp_min=0.5, temp_max=3.0):
        """
        Args:
            feature_dim: Dimension of input features (e.g., CLS token dimension)
            num_classes: Number of output classes
            temp_min: Minimum temperature
            temp_max: Maximum temperature
        """
        super().__init__()
        self.temp_predictor = TemperaturePredictor(
            feature_dim, 
            hidden_dim=feature_dim // 2,
            temp_min=temp_min,
            temp_max=temp_max
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, features, return_temperature=False):
        """
        Forward pass with instance-wise temperature scaling.
        
        Args:
            features: Input features, shape (B, feature_dim)
            return_temperature: Whether to return predicted temperatures
        
        Returns:
            logits: Calibrated logits, shape (B, num_classes)
            (optional) temperatures: Predicted temperatures, shape (B, 1)
        """
        # Predict instance-wise temperature
        temperature = self.temp_predictor(features)  # (B, 1)
        
        # Compute logits
        logits = self.classifier(features)  # (B, num_classes)
        
        # Apply temperature scaling
        calibrated_logits = logits / temperature  # (B, num_classes)
        
        if return_temperature:
            return calibrated_logits, temperature
        return calibrated_logits


def replace_attention_with_calibration(vit_model):
    """
    Replace standard attention layers in ViT with Calibration Attention.
    
    Args:
        vit_model: Vision Transformer model
    
    Returns:
        Modified model with calibration attention
    """
    # This function modifies ViT architecture
    # Implementation depends on your specific ViT architecture
    # Example for timm models:
    
    for block in vit_model.blocks:
        if hasattr(block, 'attn'):
            old_attn = block.attn
            # Replace with CalibrationAttention
            block.attn = CalibrationAttention(
                dim=old_attn.qkv.in_features,
                num_heads=old_attn.num_heads,
                qkv_bias=old_attn.qkv.bias is not None,
                attn_drop=old_attn.attn_drop.p,
                proj_drop=old_attn.proj_drop.p
            )
    
    return vit_model


# Example usage utilities

def add_calibration_to_model(model, method='temperature_scaling', **kwargs):
    """
    Add calibration to a trained model.
    
    Args:
        model: Trained classification model
        method: Calibration method ('temperature_scaling' or 'calibration_attention')
        **kwargs: Additional arguments for calibration method
    
    Returns:
        Calibrated model
    """
    if method == 'temperature_scaling':
        from .temperature_scaling import TemperatureScaling, CalibratedModel
        temp_scaler = TemperatureScaling()
        # Note: You need to call temp_scaler.fit() on validation data
        return CalibratedModel(model, temp_scaler)
    
    elif method == 'calibration_attention':
        # For ViT models
        return replace_attention_with_calibration(model)
    
    else:
        raise ValueError(f"Unknown calibration method: {method}")
