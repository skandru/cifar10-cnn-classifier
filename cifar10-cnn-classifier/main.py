"""
Main execution script for CIFAR-10 CNN Classifier

This script orchestrates the complete training pipeline:
1. Load configuration
2. Setup data loaders
3. Create model
4. Train model
5. Evaluate and visualize results
"""

import os
import sys
import yaml
from pathlib import Path
import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.model import CIFAR10CNN
from src.dataset import create_dataloaders, CIFAR10CustomDataset, get_transforms
from src.train import train_model, evaluate_model
from src.utils import (
    set_seed, get_device, save_config, 
    MetricsTracker, ModelCheckpoint
)
from src.visualize import (
    plot_training_history, visualize_predictions, 
    plot_confusion_matrix, visualize_dataset,
    plot_class_distribution, plot_model_architecture
)


def load_config(config_path='./configs/config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}, using default config")
        return get_default_config()


def get_default_config():
    """Get default configuration"""
    return {
        'model': {
            'num_classes': 10,
            'dropout_rate': 0.3
        },
        'training': {
            'num_epochs': 30,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'step_size': 15,
            'gamma': 0.1,
            'early_stopping': False,
            'patience': 10
        },
        'data': {
            'batch_size': 128,
            'num_workers': 2,
            'pin_memory': True,
            'data_root': './data',
            'download': True,
            'transforms': {
                'flip_prob': 0.5,
                'rotation_degrees': 10,
                'crop_padding': 4,
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            }
        },
        'paths': {
            'checkpoint_dir': './checkpoints',
            'results_dir': './results',
            'logs_dir': './logs'
        },
        'experiment': {
            'name': 'cifar10_cnn_baseline',
            'seed': 42,
            'save_metrics': True,
            'save_plots': True
        }
    }


def setup_directories(config):
    """Create necessary directories"""
    dirs_to_create = [
        config['paths']['checkpoint_dir'],
        config['paths']['results_dir'],
        config['paths']['logs_dir'],
        './configs'
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)


def main():
    """Main execution function"""
    print("CIFAR-10 Custom CNN Classifier")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Setup directories
    setup_directories(config)
    
    # Save configuration for reproducibility
    config_save_path = os.path.join(config['paths']['logs_dir'], 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Set random seed for reproducibility
    set_seed(config['experiment']['seed'])
    print(f"Random seed set to: {config['experiment']['seed']}")
    
    # Get device
    device = get_device()
    
    # Create data loaders
    print("\nSetting up data loaders...")
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(config['data'])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.get_class_names())}")
    print(f"Batch size: {config['data']['batch_size']}")
    
    # Visualize dataset
    if config['experiment']['save_plots']:
        print("\nGenerating dataset visualizations...")
        # Use dataset without normalization for better visualization
        raw_dataset = CIFAR10CustomDataset(train=True, transform=None)
        visualize_dataset(raw_dataset, 
                         save_path=os.path.join(config['paths']['results_dir'], 'dataset_samples.png'))
        plot_class_distribution(raw_dataset,
                               save_path=os.path.join(config['paths']['results_dir'], 'class_distribution.png'))
    
    # Create model
    print(f"\nCreating model...")
    model = CIFAR10CNN(
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    
    print(f"Model created with {model.get_num_parameters():,} parameters")
    
    # Visualize model architecture
    if config['experiment']['save_plots']:
        plot_model_architecture(model,
                               save_path=os.path.join(config['paths']['results_dir'], 'model_architecture.png'))
    
    # Train model
    print(f"\nStarting training...")
    print(f"Experiment name: {config['experiment']['name']}")
    
    # Update config with paths for training
    training_config = config['training'].copy()
    training_config.update(config['paths'])
    
    metrics_tracker = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )
    
    # Save training metrics
    if config['experiment']['save_metrics']:
        metrics_path = os.path.join(config['paths']['results_dir'], 'training_metrics.json')
        metrics_tracker.save_metrics(metrics_path)
        print(f"Training metrics saved to: {metrics_path}")
    
    # Load best model for evaluation
    checkpoint_manager = ModelCheckpoint(config['paths']['checkpoint_dir'])
    best_checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'best_checkpoint.pth')
    
    if os.path.exists(best_checkpoint_path):
        print(f"\nLoading best model for evaluation...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model loaded with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Final evaluation on test set
    print(f"\nEvaluating model on test set...")
    test_accuracy, predictions, targets = evaluate_model(model, val_loader, device)
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    
    # Generate visualizations
    if config['experiment']['save_plots']:
        print(f"\nGenerating result visualizations...")
        
        # Training history
        plot_training_history(
            metrics_tracker,
            save_path=os.path.join(config['paths']['results_dir'], 'training_history.png')
        )
        
        # Model predictions
        visualize_predictions(
            model, val_loader, device,
            save_path=os.path.join(config['paths']['results_dir'], 'predictions_visualization.png')
        )
        
        # Confusion matrix
        plot_confusion_matrix(
            model, val_loader, device,
            save_path=os.path.join(config['paths']['results_dir'], 'confusion_matrix.png')
        )
    
    # Print final summary
    print(f"\nTraining Summary")
    print("=" * 60)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Best validation accuracy: {max(metrics_tracker.val_accuracies):.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Total epochs trained: {len(metrics_tracker.train_losses)}")
    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Results saved to: {config['paths']['results_dir']}")
    print(f"Checkpoints saved to: {config['paths']['checkpoint_dir']}")
    
    print(f"\nTraining completed successfully! 🎉")


if __name__ == "__main__":
    main()