"""
Calibration methods for neural network models.
"""
from .temperature_scaling import TemperatureScaling
from .calibration_attention import CalibrationAttention

__all__ = ['TemperatureScaling', 'CalibrationAttention']
