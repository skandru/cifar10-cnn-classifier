"""
Training Pipeline for CIFAR-10 CNN

This module contains:
- Training and validation functions
- Complete training pipeline with metrics tracking
- Learning rate scheduling and optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .utils import MetricsTracker, ModelCheckpoint


def train_epoch(model, train_loader, criterion, optimizer, device, epoch=None):
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        epoch: Current epoch number (for display)
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc = f"Training Epoch {epoch}" if epoch is not None else "Training"
    train_bar = tqdm(train_loader, desc=desc, leave=False)
    
    for batch_idx, (data, targets) in enumerate(train_bar):
        data, targets = data.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Update progress bar
        if batch_idx % 50 == 0:  # Update every 50 batches
            train_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device, epoch=None):
    """
    Validate model for one epoch
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
        epoch: Current epoch number (for display)
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc = f"Validation Epoch {epoch}" if epoch is not None else "Validation"
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=desc, leave=False)
        
        for batch_idx, (data, targets) in enumerate(val_bar):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            if batch_idx % 50 == 0:
                val_bar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, config, device):
    """
    Complete training pipeline
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary with training parameters
        device: Device to train on
        
    Returns:
        MetricsTracker: Object containing training history
    """
    
    # Training configuration
    num_epochs = config.get('num_epochs', 50)
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-4)
    step_size = config.get('step_size', 15)
    gamma = config.get('gamma', 0.1)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Initialize tracking
    metrics_tracker = MetricsTracker()
    checkpoint_manager = ModelCheckpoint(config.get('checkpoint_dir', './checkpoints'))
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Model has {model.get_num_parameters():,} trainable parameters")
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch+1
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track metrics
        metrics_tracker.update(train_loss, train_acc, val_loss, val_acc, current_lr)
        
        # Save checkpoint
        is_best = val_acc > checkpoint_manager.best_val_acc
        checkpoint_manager.save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss, val_acc, is_best
        )
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Early stopping check (optional)
        if config.get('early_stopping', False):
            patience = config.get('patience', 10)
            if checkpoint_manager.should_stop(patience):
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {checkpoint_manager.best_val_acc:.2f}%")
    
    return metrics_tracker


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        tuple: (test_accuracy, predictions, targets)
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Evaluating")
        
        for data, targets in test_bar:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            test_bar.set_postfix({'Acc': f'{100.*correct/total:.2f}%'})
    
    test_accuracy = 100. * correct / total
    return test_accuracy, all_predictions, all_targets


if __name__ == "__main__":
    # Example usage
    from .model import CIFAR10CNN
    from .dataset import create_dataloaders
    
    # Configuration
    config = {
        'num_epochs': 5,
        'learning_rate': 0.001,
        'batch_size': 128,
        'data_root': './data'
    }
    
    # Create model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CIFAR10CNN().to(device)
    train_loader, val_loader, _, _ = create_dataloaders(config)
    
    # Train model
    metrics = train_model(model, train_loader, val_loader, config, device)
    print("Training completed successfully!")