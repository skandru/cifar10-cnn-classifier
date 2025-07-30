"""
Visualization Tools for CIFAR-10 CNN Project

This module contains:
- Training history plots
- Model predictions visualization
- Confusion matrix generation
- Dataset exploration plots
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os

from .dataset import CIFAR10_CLASSES, denormalize_tensor


def plot_training_history(metrics_tracker, save_path='./results/training_history.png'):
    """
    Plot comprehensive training history
    
    Args:
        metrics_tracker: MetricsTracker object with training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = metrics_tracker.epochs if metrics_tracker.epochs else range(1, len(metrics_tracker.train_losses) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, metrics_tracker.train_losses, 'bo-', label='Training Loss', alpha=0.7)
    axes[0, 0].plot(epochs, metrics_tracker.val_losses, 'ro-', label='Validation Loss', alpha=0.7)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, metrics_tracker.train_accuracies, 'bo-', label='Training Accuracy', alpha=0.7)
    axes[0, 1].plot(epochs, metrics_tracker.val_accuracies, 'ro-', label='Validation Accuracy', alpha=0.7)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(epochs, metrics_tracker.learning_rates, 'go-', alpha=0.7)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance comparison
    best_val_acc = max(metrics_tracker.val_accuracies) if metrics_tracker.val_accuracies else 0
    axes[1, 1].plot(epochs, metrics_tracker.val_accuracies, 'ro-', label='Validation Accuracy', alpha=0.7)
    axes[1, 1].axhline(y=best_val_acc, color='g', linestyle='--', 
                       label=f'Best: {best_val_acc:.2f}%', alpha=0.8)
    axes[1, 1].set_title('Validation Performance', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.show()


def visualize_predictions(model, test_loader, device, num_samples=16, save_path='./results/predictions_visualization.png'):
    """
    Visualize model predictions on test samples
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
    """
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.ravel() if num_samples > 1 else [axes]
    
    for i in range(min(num_samples, len(images))):
        # Denormalize image for display
        img = denormalize_tensor(images[i].cpu())
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img.permute(1, 2, 0))
        
        true_label = CIFAR10_CLASSES[labels[i]]
        pred_label = CIFAR10_CLASSES[predicted[i]]
        confidence = probabilities[i][predicted[i]].item()
        
        color = 'green' if predicted[i] == labels[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
                         color=color, fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Model Predictions on Test Set', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Predictions visualization saved to {save_path}")
    plt.show()


def plot_confusion_matrix(model, test_loader, device, save_path='./results/confusion_matrix.png'):
    """
    Generate and plot confusion matrix
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to run inference on
        save_path: Path to save the plot
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    print("Computing predictions for confusion matrix...")
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc='Computing confusion matrix'):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES,
                cbar_kws={'label': 'Number of Samples'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.show()
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print("=" * 60)
    print(classification_report(all_labels, all_predictions, target_names=CIFAR10_CLASSES))
    
    return cm


def visualize_dataset(dataset, num_samples=12, save_path='./results/dataset_samples.png'):
    """
    Visualize random samples from the dataset
    
    Args:
        dataset: CIFAR-10 dataset
        num_samples: Number of samples to display
        save_path: Path to save the plot
    """
    # Calculate grid size
    grid_rows = 3
    grid_cols = 4
    num_samples = min(num_samples, grid_rows * grid_cols)
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 9))
    axes = axes.ravel()
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # Handle different image formats
        if isinstance(image, torch.Tensor):
            # If normalized, denormalize for display
            if image.min() < 0:  # Likely normalized
                img = denormalize_tensor(image)
                img = torch.clamp(img, 0, 1)
                img = img.numpy().transpose((1, 2, 0))
            else:
                img = image.numpy().transpose((1, 2, 0))
        else:
            img = np.array(image) / 255.0
        
        axes[i].imshow(img)
        axes[i].set_title(f'{CIFAR10_CLASSES[label]}', fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('CIFAR-10 Dataset Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Dataset samples saved to {save_path}")
    plt.show()


def plot_class_distribution(dataset, save_path='./results/class_distribution.png'):
    """
    Plot class distribution in the dataset
    
    Args:
        dataset: CIFAR-10 dataset
        save_path: Path to save the plot
    """
    distribution = dataset.get_class_distribution()
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(CIFAR10_CLASSES, distribution, color='skyblue', alpha=0.7, edgecolor='navy')
    
    # Add value labels on bars
    for bar, count in zip(bars, distribution):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Class Distribution in CIFAR-10 Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Class distribution saved to {save_path}")
    plt.show()


def plot_model_architecture(model, save_path='./results/model_architecture.png'):
    """
    Visualize model architecture summary
    
    Args:
        model: PyTorch model
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create architecture text
    arch_text = f"""
    CIFAR-10 CNN Architecture
    
    Total Parameters: {total_params:,}
    Trainable Parameters: {trainable_params:,}
    
    Architecture:
    ├── Conv Block 1 (3→64 channels)
    │   ├── Conv2d(3, 64, 3x3) + BatchNorm + ReLU
    │   ├── Conv2d(64, 64, 3x3) + BatchNorm + ReLU
    │   ├── MaxPool2d(2x2) + Dropout2d
    │   └── Output: 64 × 16 × 16
    │
    ├── Conv Block 2 (64→128 channels)
    │   ├── Conv2d(64, 128, 3x3) + BatchNorm + ReLU
    │   ├── Conv2d(128, 128, 3x3) + BatchNorm + ReLU
    │   ├── MaxPool2d(2x2) + Dropout2d
    │   └── Output: 128 × 8 × 8
    │
    ├── Conv Block 3 (128→256 channels)
    │   ├── Conv2d(128, 256, 3x3) + BatchNorm + ReLU
    │   ├── Conv2d(256, 256, 3x3) + BatchNorm + ReLU
    │   ├── MaxPool2d(2x2) + Dropout2d
    │   └── Output: 256 × 4 × 4
    │
    └── Classifier
        ├── Linear(4096, 512) + ReLU + Dropout
        ├── Linear(512, 256) + ReLU + Dropout
        └── Linear(256, 10)
    """
    
    ax.text(0.05, 0.95, arch_text, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model architecture saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    # Test visualization functions
    from .dataset import CIFAR10CustomDataset, get_transforms
    
    print("Testing visualization functions...")
    
    # Test dataset visualization
    _, val_transform = get_transforms()
    dataset = CIFAR10CustomDataset(train=False, transform=val_transform)
    
    print("Creating dataset samples visualization...")
    visualize_dataset(dataset, num_samples=12)
    
    print("Creating class distribution plot...")
    plot_class_distribution(dataset)
    
    print("Visualization tests completed!")