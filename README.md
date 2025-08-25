SwinIR Super-Resolution Block on CIFAR-10

This repository contains an implementation of a SwinIR-like transformer block for image super-resolution/enhancement. The model is tested on CIFAR-10 images to demonstrate its functionality.

Features

SwinIR Block: A lightweight Swin Transformer-based residual block with windowed self-attention.

Window Attention: Local attention within windows with optional shifted windows.

Residual MLP: Feature refinement with residual connections.

Input/Output: Takes RGB images and outputs enhanced RGB images of the same resolution.

PyTorch Implementation: Fully implemented using torch and torchvision.

Repository Structure
swinir-cifar10/
├── swinir.py          # SwinIR block implementation
├── test_swinir.py     # Testing script on CIFAR-10 dataset
├── README.md

Installation

Clone the repository:

git clone <repository-url>
cd swinir-cifar10


Install dependencies:

pip install torch torchvision matplotlib

Usage
Testing SwinIR on CIFAR-10

Run the test script:

python test_swinir.py


This script:

Loads a batch of CIFAR-10 images.

Passes them through the SwinIRBlock.

Visualizes the input vs output images.

Prints the input and output tensor shapes.

Parameters

You can modify the SwinIR block parameters in test_swinir.py:

dim: Feature dimension of the block (default: 96)

input_resolution: Input image size (default: 32x32 for CIFAR-10)

num_heads: Number of attention heads (default: 3)

window_size: Size of local attention window (default: 8)

shift_size: Shift size for window attention (default: 4)

Visualization

The visualize function shows the first 6 input images alongside their enhanced outputs.

Notes

This is a single SwinIR block demo and not a full SwinIR network.

For super-resolution tasks on higher-resolution datasets, multiple blocks can be stacked.

CIFAR-10 is used here as a minimal test dataset.

License

MIT License
