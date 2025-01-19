#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from autoencoder01 import Autoencoder

# Set random seed for reproducibility
torch.manual_seed(1453)

# Check for available device
device = torch.device("cuda" if torch.cuda.is_available() 
                     else "mps" if torch.backends.mps.is_available() 
                     else "cpu")

def create_experiment_dir():
    # Create experiments directory if it doesn't exist
    experiments_dir = 'experiments'
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    
    # Create timestamp-based experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(experiments_dir, f'experiment_{timestamp}')
    
    # Create subdirectories
    subdirs = ['checkpoints', 'plots', 'logs', 'models']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir))
    
    return experiment_dir

def save_experiment_config(experiment_dir, config):
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def create_dataloaders(batch_size=128):
    # Define the transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                   download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                  download=True, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, data)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}]: Loss {loss.item():.6f}')
    
    return total_loss / len(train_loader)

def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def save_training_plot(train_losses, val_losses, experiment_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, 'plots', 'training_history.png'))
    plt.close()

def main():
    # Hyperparameters and configuration
    config = {
        'input_shape': (3, 32, 32),
        'batch_size': 128,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'device': str(device),
        'model_type': 'Autoencoder',
        'dataset': 'CIFAR-10',
        'optimizer': 'Adam',
        'loss_function': 'MSE',
    }
    
    # Create experiment directory
    experiment_dir = create_experiment_dir()
    
    # Save experiment configuration
    save_experiment_config(experiment_dir, config)
    
    # Create model and move to device
    model = Autoencoder(config['input_shape']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(config['batch_size'])
    
    # Training loop
    train_losses = []
    val_losses = []
    
    # Create log file
    log_file = os.path.join(experiment_dir, 'logs', 'training_log.txt')
    
    print(f"Training on {device}")
    print(f"Experiment directory: {experiment_dir}")
    
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training on {device}\n\n")
        
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss = validate(model, test_loader, criterion, device)
            
            # Log results
            log_message = f"Epoch {epoch+1}/{config['num_epochs']}\n"
            log_message += f"Training Loss: {train_loss:.6f}\n"
            log_message += f"Validation Loss: {val_loss:.6f}\n"
            
            print(log_message)
            f.write(log_message + '\n')
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Save model checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    experiment_dir, 
                    'checkpoints', 
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(experiment_dir, 'models', 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # Save training history plot
    save_training_plot(train_losses, val_losses, experiment_dir)
    
    # Save training history data
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    torch.save(history, os.path.join(experiment_dir, 'logs', 'training_history.pth'))

if __name__ == "__main__":
    main()
