#!/usr/bin/env python3


# import required libraries
import torch

# check which device is available
if torch.backends.mps.is_available():
    backend_device = torch.device("mps")
    print ("MPS is available.")
elif torch.backends.cuda.is_available():
    backend_device = torch.device("cuda")
    print ("CUDA is available.")
elif torch.backends.cpu.is_available():
    backend_device = torch.device("cpu")
    print ("CPU is available.")
else:
    print ("None of the devices are available!")

x = torch.rand(5, 3, device=backend_device)
print(x)
