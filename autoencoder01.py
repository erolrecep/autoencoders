#!/usr/bin/env python3

import torch
import torch.nn as nn
from torchviz import make_dot


class Encoder(nn.Module):
    def __init__(self, input_shape):
        super(Encoder, self).__init__()
        
        # Calculate dimensions based on input shape
        # CIFAR-10 images are (3, 32, 32)
        channels, height, width = input_shape
        
        self.encoder = nn.Sequential(
            # First convolution block
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Third convolution block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, output_shape):
        super(Decoder, self).__init__()
        
        channels, height, width = output_shape
        
        self.decoder = nn.Sequential(
            # First deconvolution block
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Second deconvolution block
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Final convolution to get back to original channels
            nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Sigmoid for pixel values between 0 and 1
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, input_shape=(3, 32, 32)):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_shape)
        self.decoder = Decoder(input_shape)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_encoded_size(self, input_shape):
        # Helper method to calculate the size of encoded representation
        x = torch.randn(1, *input_shape)
        encoded = self.encoder(x)
        return encoded.shape[1:]


if __name__ == "__main__":
    # Test the autoencoder with CIFAR-10 dimensions
    input_shape = (3, 32, 32)
    model = Autoencoder(input_shape)
    
    # Print model summary
    x = torch.randn(1, *input_shape)
    encoded = model.encoder(x)
    decoded = model.decoder(encoded)
    
    print(f"Input shape: {x.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Decoded shape: {decoded.shape}")

    # Visualize the model using torchviz
    make_dot(encoded, params=dict(list(model.named_parameters()))).render("output/encoder", format="png")
    make_dot(decoded, params=dict(list(model.named_parameters()))).render("output/decoder", format="png")
