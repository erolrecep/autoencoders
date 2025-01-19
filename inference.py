#!/usr/bin/env python3

import torch
from torchvision import transforms
from PIL import Image
import os
import json
from datetime import datetime
import glob
from autoencoder01 import Autoencoder
import matplotlib.pyplot as plt
import numpy as np

def list_experiments():
    experiments_dir = 'experiments'
    if not os.path.exists(experiments_dir):
        print("No experiments found!")
        return None
    
    experiments = []
    for exp_dir in sorted(glob.glob(os.path.join(experiments_dir, 'experiment_*'))):
        config_path = os.path.join(exp_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get training log details
            log_path = os.path.join(exp_dir, 'logs', 'training_log.txt')
            training_date = "Unknown"
            final_loss = "Unknown"
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        training_date = lines[0].strip().split('at ')[-1]
                        for line in reversed(lines):
                            if 'Validation Loss:' in line:
                                final_loss = float(line.split(':')[-1].strip())
                                break
            
            experiments.append({
                'path': exp_dir,
                'date': training_date,
                'config': config,
                'final_loss': final_loss
            })
    
    return experiments

def select_experiment(experiments):
    print("\nAvailable experiments:")
    print("-" * 80)
    print(f"{'Index':<6} {'Date':<20} {'Final Loss':<15} {'Device':<8} {'Epochs':<8}")
    print("-" * 80)
    
    for idx, exp in enumerate(experiments):
        print(f"{idx:<6} {exp['date']:<20} {exp['final_loss']:<15.6f} "
              f"{exp['config']['device']:<8} {exp['config']['num_epochs']:<8}")
    
    while True:
        try:
            choice = int(input("\nSelect experiment index: "))
            if 0 <= choice < len(experiments):
                return experiments[choice]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

def load_model(experiment_path, config):
    model = Autoencoder(tuple(config['input_shape']))
    model_path = os.path.join(experiment_path, 'models', 'final_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def process_image(image_path, model, device, input_shape):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((input_shape[1], input_shape[2])),
        transforms.ToTensor(),
    ])
    
    # Load and transform image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Process image
    with torch.no_grad():
        output = model(image_tensor)
    
    return image_tensor, output

def save_results(original_tensor, reconstructed_tensor, experiment_path):
    # Create results directory
    results_dir = os.path.join(experiment_path, 'inference_results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert tensors to images
    def tensor_to_image(tensor):
        img = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        return img
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original image
    ax1.imshow(tensor_to_image(original_tensor))
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Plot reconstructed image
    ax2.imshow(tensor_to_image(reconstructed_tensor))
    ax2.set_title('Reconstructed')
    ax2.axis('off')
    
    # Save figure
    plt.savefig(os.path.join(results_dir, f'comparison_{timestamp}.png'))
    plt.close()
    
    return os.path.join(results_dir, f'comparison_{timestamp}.png')

def main():
    # List available experiments
    experiments = list_experiments()
    if not experiments:
        return
    
    # Let user select an experiment
    selected_exp = select_experiment(experiments)
    print(f"\nSelected experiment from: {selected_exp['date']}")
    
    try:
        # Load model
        model, device = load_model(selected_exp['path'], selected_exp['config'])
        print(f"Model loaded successfully. Using device: {device}")
        
        # Get image path from user
        while True:
            image_path = input("\nEnter path to image file (or 'q' to quit): ")
            if image_path.lower() == 'q':
                break
                
            if not os.path.exists(image_path):
                print("Image file not found. Please try again.")
                continue
            
            try:
                # Process image
                original, reconstructed = process_image(
                    image_path, 
                    model, 
                    device, 
                    selected_exp['config']['input_shape']
                )
                
                # Save and display results
                result_path = save_results(original, reconstructed, selected_exp['path'])
                print(f"\nResults saved to: {result_path}")
                
                # Calculate and display compression ratio
                original_size = os.path.getsize(image_path)
                compressed_size = os.path.getsize(result_path)
                compression_ratio = original_size / compressed_size
                print(f"Compression ratio: {compression_ratio:.2f}x")
                
            except Exception as e:
                print(f"Error processing image: {str(e)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
