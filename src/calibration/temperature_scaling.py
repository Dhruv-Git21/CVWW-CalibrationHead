"""
Temperature Scaling for model calibration.

Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling calibration method.
    
    Applies a single scalar temperature parameter T to the logits:
        p_i = exp(z_i/T) / sum_j exp(z_j/T)
    
    The temperature T is learned on a validation set by minimizing NLL.
    """
    
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        # Initialize temperature as a learnable parameter
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        """
        Scale logits by temperature.
        
        Args:
            logits: Raw model outputs before softmax, shape (batch_size, num_classes)
        
        Returns:
            Scaled logits: logits / T
        """
        return logits / self.temperature
    
    def fit(self, model, val_loader, device='cuda', max_iter=50, lr=0.01):
        """
        Learn the temperature parameter on validation set.
        
        Args:
            model: Trained classification model
            val_loader: Validation data loader
            device: Device to use for computation
            max_iter: Maximum optimization iterations
            lr: Learning rate for temperature optimization
        
        Returns:
            Optimized temperature value
        """
        model.eval()
        self.to(device)
        
        # Disable gradient computation for model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Collect all logits and labels
        logits_list = []
        labels_list = []
        
        print("Collecting logits from validation set...")
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                logits_list.append(logits.cpu())
                labels_list.append(labels)
        
        # Concatenate all batches
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        
        # Use grid search to find optimal temperature (more robust than LBFGS)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Searching for optimal temperature using grid search...")
        
        # Grid search over temperature range
        temperatures = torch.linspace(0.5, 5.0, 50)
        best_loss = float('inf')
        best_temp = 1.5
        
        nll_criterion = nn.NLLLoss()
        
        for temp in temperatures:
            self.temperature.data = torch.tensor([temp])
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_temp = temp.item()
        
        # Set to best temperature
        self.temperature.data = torch.tensor([best_temp])
        final_temp = best_temp
        
        print(f"Optimal temperature found: T={final_temp:.4f} (NLL={best_loss:.4f})")
        
        return final_temp
    
    def calibrate_model(self, model):
        """
        Wrap a model with temperature scaling.
        
        Args:
            model: Original model
        
        Returns:
            CalibratedModel that applies temperature scaling
        """
        return CalibratedModel(model, self)


class CalibratedModel(nn.Module):
    """
    Wrapper that applies temperature scaling to a model.
    """
    
    def __init__(self, model, temperature_module):
        super(CalibratedModel, self).__init__()
        self.model = model
        self.temperature = temperature_module.temperature
    
    def forward(self, x):
        """Forward pass with temperature scaling."""
        logits = self.model(x)
        return logits / self.temperature


def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between confidence and accuracy:
        ECE = sum_m (|B_m|/n) * |acc(B_m) - conf(B_m)|
    
    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)
        n_bins: Number of bins for binning predictions
    
    Returns:
        ECE score (lower is better)
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Find samples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def compute_nll(probs, labels):
    """
    Compute Negative Log-Likelihood (NLL).
    
    NLL = -mean(log p(y_true | x))
    
    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)
    
    Returns:
        NLL score (lower is better)
    """
    # Get probabilities of true class
    true_probs = probs[torch.arange(len(labels)), labels]
    # Compute negative log-likelihood
    nll = -torch.log(true_probs + 1e-8).mean()
    return nll.item()


def evaluate_calibration(model, data_loader, device='cuda', n_bins=15):
    """
    Evaluate model calibration metrics.
    
    Args:
        model: Classification model
        data_loader: Data loader for evaluation
        device: Device to use
        n_bins: Number of bins for ECE computation
    
    Returns:
        Dictionary with calibration metrics:
            - accuracy: Classification accuracy
            - ece: Expected Calibration Error
            - nll: Negative Log-Likelihood
    """
    model.eval()
    all_probs = []
    all_labels = []
    correct = 0
    total = 0
    
    print(f"Evaluating on {len(data_loader)} batches...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches...")
    
    print("Computing metrics...")
    # Concatenate all batches
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    # Compute metrics
    accuracy = 100. * correct / total
    ece = compute_ece(all_probs, all_labels, n_bins)
    nll = compute_nll(all_probs, all_labels)
    
    return {
        'accuracy': accuracy,
        'ece': ece,
        'nll': nll
    }


def plot_reliability_diagram(probs, labels, n_bins=10, save_path=None):
    """
    Plot reliability diagram (calibration curve).
    
    X-axis: Predicted confidence
    Y-axis: Actual accuracy
    Perfect calibration = diagonal line
    
    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)
        n_bins: Number of bins
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt
    
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum().item() > 0:
            bin_confidences.append(confidences[in_bin].mean().item())
            bin_accuracies.append(accuracies[in_bin].float().mean().item())
            bin_counts.append(in_bin.sum().item())
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bar plot
    ax.bar(range(len(bin_accuracies)), bin_accuracies, 
           width=0.8, alpha=0.7, label='Accuracy', color='steelblue')
    
    # Confidence line
    ax.plot(range(len(bin_confidences)), bin_confidences, 
            'ro-', linewidth=2, label='Confidence', markersize=8)
    
    # Perfect calibration line
    ax.plot([0, len(bin_accuracies)-1], [0, 1], 
            'k--', linewidth=2, alpha=0.5, label='Perfect Calibration')
    
    ax.set_xlabel('Confidence Bin', fontsize=12)
    ax.set_ylabel('Accuracy / Confidence', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Add bin counts as text
    for i, count in enumerate(bin_counts):
        ax.text(i, 0.05, f'n={count}', ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reliability diagram saved to {save_path}")
    
    plt.show()
    return fig
