"""
Metrics utilities for tracking and logging.
"""
import csv
import os
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class MetricsLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str, resume: bool = False):
        """
        Args:
            log_dir: Directory to save logs
            resume: Whether to resume logging from existing file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.log_dir / 'metrics.csv'
        self.metrics_history = []
        
        if resume and self.csv_path.exists():
            self._load_history()
        else:
            self.fieldnames = None
    
    def _load_history(self):
        """Load metrics history from CSV."""
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            self.fieldnames = reader.fieldnames
            for row in reader:
                # Convert numeric values
                row_converted = {}
                for k, v in row.items():
                    try:
                        row_converted[k] = float(v)
                    except (ValueError, TypeError):
                        row_converted[k] = v
                self.metrics_history.append(row_converted)
    
    def log(self, metrics: Dict, epoch: Optional[int] = None):
        """
        Log metrics for current epoch.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Epoch number (optional)
        """
        if epoch is not None:
            metrics['epoch'] = epoch
        
        self.metrics_history.append(metrics)
        
        # Set fieldnames on first log
        if self.fieldnames is None:
            self.fieldnames = list(metrics.keys())
        
        # Write to CSV
        file_exists = self.csv_path.exists()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
    
    def get_best_metric(self, metric_name: str, mode: str = 'max') -> float:
        """
        Get best value of a metric.
        
        Args:
            metric_name: Name of metric
            mode: 'max' or 'min'
        
        Returns:
            Best metric value
        """
        if not self.metrics_history:
            return float('-inf') if mode == 'max' else float('inf')
        
        values = [m.get(metric_name, float('-inf' if mode == 'max' else 'inf')) 
                  for m in self.metrics_history]
        
        if mode == 'max':
            return max(values)
        else:
            return min(values)
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """
        Plot training metrics.
        
        Args:
            save_path: Path to save plot (optional)
        """
        if not self.metrics_history:
            return
        
        # Extract epochs and metrics
        epochs = [m.get('epoch', i) for i, m in enumerate(self.metrics_history)]
        
        # Identify train and val metrics
        train_metrics = [k for k in self.fieldnames if k.startswith('train_')]
        val_metrics = [k for k in self.fieldnames if k.startswith('val_')]
        
        if not train_metrics and not val_metrics:
            # No train/val prefix, plot all numeric metrics
            numeric_metrics = []
            for k in self.fieldnames:
                if k != 'epoch' and isinstance(self.metrics_history[0].get(k), (int, float)):
                    numeric_metrics.append(k)
            train_metrics = numeric_metrics
        
        # Create subplots
        num_plots = max(len(train_metrics), len(val_metrics))
        if num_plots == 0:
            return
        
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        
        for idx, metric in enumerate(train_metrics):
            ax = axes[idx]
            
            # Plot train metric
            train_values = [m.get(metric, None) for m in self.metrics_history]
            if any(v is not None for v in train_values):
                ax.plot(epochs, train_values, label=f'Train {metric}', marker='o')
            
            # Plot val metric if exists
            val_metric = metric.replace('train_', 'val_')
            if val_metric in self.fieldnames:
                val_values = [m.get(val_metric, None) for m in self.metrics_history]
                if any(v is not None for v in val_values):
                    ax.plot(epochs, val_values, label=f'Val {val_metric}', marker='s')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.log_dir / 'metrics_plot.png', dpi=150, bbox_inches='tight')
        
        plt.close()


class EMAMeter:
    """Exponential Moving Average meter."""
    
    def __init__(self, alpha: float = 0.9):
        """
        Args:
            alpha: Smoothing factor (0-1)
        """
        self.alpha = alpha
        self.value = None
    
    def update(self, val: float):
        """Update EMA with new value."""
        if self.value is None:
            self.value = val
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * val
    
    def get(self) -> float:
        """Get current EMA value."""
        return self.value if self.value is not None else 0.0


def format_metrics(metrics: Dict, prefix: str = '') -> str:
    """
    Format metrics dictionary as string.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for metric names
    
    Returns:
        Formatted string
    """
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f'{prefix}{k}: {v:.4f}')
        else:
            parts.append(f'{prefix}{k}: {v}')
    return ', '.join(parts)


def print_metrics_summary(train_metrics: Dict, val_metrics: Optional[Dict] = None, epoch: int = 0):
    """
    Print formatted metrics summary.
    
    Args:
        train_metrics: Training metrics
        val_metrics: Validation metrics (optional)
        epoch: Current epoch
    """
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train: {format_metrics(train_metrics)}")
    if val_metrics:
        print(f"  Val:   {format_metrics(val_metrics)}")
