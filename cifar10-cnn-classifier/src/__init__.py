"""
CIFAR-10 CNN Classifier Package
A comprehensive PyTorch implementation for CIFAR-10 image classification
"""

__version__ = "1.0.0"
__author__ = "Suresh Kandru"

from .model import CIFAR10CNN
from .dataset import CIFAR10CustomDataset, get_transforms
from .train import train_model, train_epoch, validate_epoch
from .utils import MetricsTracker, ModelCheckpoint, set_seed
from .visualize import (
    plot_training_history, 
    visualize_predictions, 
    plot_confusion_matrix,
    visualize_dataset
)

__all__ = [
    'CIFAR10CNN',
    'CIFAR10CustomDataset', 
    'get_transforms',
    'train_model',
    'train_epoch',
    'validate_epoch',
    'MetricsTracker',
    'ModelCheckpoint',
    'set_seed',
    'plot_training_history',
    'visualize_predictions',
    'plot_confusion_matrix',
    'visualize_dataset'
]