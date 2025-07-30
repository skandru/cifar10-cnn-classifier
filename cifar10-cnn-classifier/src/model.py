"""
CNN Architecture for CIFAR-10 Classification

This module contains the custom CNN architecture with:
- 3 Convolutional blocks with BatchNorm and Dropout
- Progressive channel scaling (64→128→256)
- Fully connected classifier with regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """
    Custom CNN Architecture for CIFAR-10 Classification
    
    Architecture:
    - 3 Convolutional Blocks with BatchNorm and Dropout
    - Each block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout
    - Fully Connected layers with Dropout
    - Output layer with 10 classes
    
    Args:
        num_classes (int): Number of output classes (default: 10)
        dropout_rate (float): Dropout probability (default: 0.3)
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(CIFAR10CNN, self).__init__()
        
        # First Convolutional Block (32x32 -> 16x16)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Second Convolutional Block (16x16 -> 8x8)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Third Convolutional Block (8x8 -> 4x4)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def create_model(config):
    """
    Factory function to create model from configuration
    
    Args:
        config (dict): Configuration dictionary with model parameters
        
    Returns:
        CIFAR10CNN: Initialized model
    """
    return CIFAR10CNN(
        num_classes=config.get('num_classes', 10),
        dropout_rate=config.get('dropout_rate', 0.3)
    )


if __name__ == "__main__":
    # Test model creation and forward pass
    model = CIFAR10CNN()
    print(f"Model created with {model.get_num_parameters():,} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")