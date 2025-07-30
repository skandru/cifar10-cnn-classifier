"""
Utility Functions for CIFAR-10 CNN Project

This module contains:
- Metrics tracking
- Model checkpointing
- Helper functions
"""

import torch
import os
import json
import random
import numpy as np
from datetime import datetime


def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricsTracker:
    """
    Track and store training metrics throughout training
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epochs = []
        
    def update(self, train_loss, train_acc, val_loss, val_acc, lr, epoch=None):
        """
        Update metrics with latest values
        
        Args:
            train_loss (float): Training loss
            train_acc (float): Training accuracy
            val_loss (float): Validation loss
            val_acc (float): Validation accuracy
            lr (float): Current learning rate
            epoch (int): Current epoch number
        """
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        
        if epoch is not None:
            self.epochs.append(epoch)
        else:
            self.epochs.append(len(self.train_losses))
    
    def get_best_epoch(self):
        """Get epoch with best validation accuracy"""
        if not self.val_accuracies:
            return None
        best_idx = np.argmax(self.val_accuracies)
        return self.epochs[best_idx], self.val_accuracies[best_idx]
    
    def save_metrics(self, filepath):
        """Save metrics to JSON file"""
        metrics_dict = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'epochs': self.epochs,
            'best_epoch': self.get_best_epoch(),
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def load_metrics(self, filepath):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            metrics_dict = json.load(f)
        
        self.train_losses = metrics_dict['train_losses']
        self.train_accuracies = metrics_dict['train_accuracies']
        self.val_losses = metrics_dict['val_losses']
        self.val_accuracies = metrics_dict['val_accuracies']
        self.learning_rates = metrics_dict['learning_rates']
        self.epochs = metrics_dict['epochs']


class ModelCheckpoint:
    """
    Handle model checkpointing and saving
    """
    
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss, val_acc, is_best=False):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: Optimizer state
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            val_acc: Validation accuracy
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            self.best_val_acc = val_acc
            self.epochs_without_improvement = 0
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            self.epochs_without_improvement += 1
    
    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            checkpoint_path: Path to checkpoint file
            
        Returns:
            tuple: (epoch, val_acc)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        return checkpoint['epoch'], checkpoint['val_acc']
    
    def should_stop(self, patience):
        """
        Check if training should stop due to early stopping
        
        Args:
            patience: Number of epochs to wait without improvement
            
        Returns:
            bool: Whether to stop training
        """
        return self.epochs_without_improvement >= patience


def count_parameters(model):
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_device():
    """
    Get the best available device
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def save_config(config, filepath):
    """
    Save configuration to JSON file
    
    Args:
        config (dict): Configuration dictionary
        filepath (str): Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(filepath):
    """
    Load configuration from JSON file
    
    Args:
        filepath (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving
    """
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test seed setting
    set_seed(42)
    print("Random seed set to 42")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update(0.5, 85.0, 0.6, 82.0, 0.001)
    print(f"Metrics updated: {len(tracker.train_losses)} entries")