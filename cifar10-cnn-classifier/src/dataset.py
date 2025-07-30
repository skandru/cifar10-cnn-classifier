"""
Custom Dataset and Data Transforms for CIFAR-10

This module handles:
- Custom CIFAR-10 dataset with advanced transforms
- Data augmentation strategies
- Normalization and preprocessing
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CIFAR-10 normalization values (computed from training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class CIFAR10CustomDataset(Dataset):
    """
    Custom CIFAR-10 Dataset with advanced transforms and augmentation
    
    Args:
        root (str): Root directory for data storage
        train (bool): If True, creates training dataset, else test dataset
        transform (callable): Transform to apply to images
        download (bool): If True, downloads dataset if not found
    """
    
    def __init__(self, root='./data', train=True, transform=None, download=True):
        self.cifar10 = CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.train = train
        
    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        image, label = self.cifar10[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_class_distribution(self):
        """Get distribution of classes in the dataset"""
        labels = [self.cifar10[i][1] for i in range(len(self.cifar10))]
        return np.bincount(labels)
    
    def get_class_names(self):
        """Get CIFAR-10 class names"""
        return CIFAR10_CLASSES


def get_transforms(config=None):
    """
    Create training and validation transforms
    
    Args:
        config (dict): Configuration dictionary with transform parameters
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    if config is None:
        config = {}
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=config.get('flip_prob', 0.5)),
        transforms.RandomRotation(config.get('rotation_degrees', 10)),
        transforms.RandomCrop(32, padding=config.get('crop_padding', 4)),
        transforms.ColorJitter(
            brightness=config.get('brightness', 0.2),
            contrast=config.get('contrast', 0.2),
            saturation=config.get('saturation', 0.2),
            hue=config.get('hue', 0.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    return train_transform, val_transform


def denormalize_tensor(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """
    Denormalize a tensor for visualization
    
    Args:
        tensor (torch.Tensor): Normalized tensor
        mean (tuple): Mean values used for normalization
        std (tuple): Standard deviation values used for normalization
        
    Returns:
        torch.Tensor: Denormalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def create_dataloaders(config):
    """
    Create training and validation dataloaders
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset)
    """
    # Get transforms
    train_transform, val_transform = get_transforms(config.get('transforms', {}))
    
    # Create datasets
    train_dataset = CIFAR10CustomDataset(
        root=config.get('data_root', './data'),
        train=True,
        transform=train_transform,
        download=config.get('download', True)
    )
    
    val_dataset = CIFAR10CustomDataset(
        root=config.get('data_root', './data'),
        train=False,
        transform=val_transform,
        download=config.get('download', True)
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        pin_memory=config.get('pin_memory', True)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=config.get('pin_memory', True)
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == "__main__":
    # Test dataset creation
    train_transform, val_transform = get_transforms()
    
    train_dataset = CIFAR10CustomDataset(train=True, transform=train_transform)
    val_dataset = CIFAR10CustomDataset(train=False, transform=val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.get_class_names()}")
    
    # Test class distribution
    distribution = train_dataset.get_class_distribution()
    for i, count in enumerate(distribution):
        print(f"{CIFAR10_CLASSES[i]}: {count} samples")