# PyTorch Deep Learning Models

This project showcases implementations of fundamental deep learning architectures using PyTorch, including Convolutional Neural Networks (CNNs), Residual Networks (ResNets) for MNIST digit classification, and the Multi-Head Attention mechanism.

## Features

-   **CNN.py**: A basic Convolutional Neural Network model definition and a simple test demonstrating its forward pass and parameter count.
-   **CNNMNIST.py**: A complete implementation of a ResNet-like model for MNIST classification, featuring data loading, augmentations, training, and evaluation loops with mixed precision training support.
-   **attention.py**: Provides the core `attention` function and a `MultiHeadAttention` module, essential components for Transformer architectures.

## Technologies Used

-   Python
-   PyTorch
-   NumPy
-   torchvision

## How to Run

First, ensure you have the necessary libraries installed:

 pip install torch torchvision numpy

### Running `CNN.py`

This script defines a simple CNN and prints its output shape and total parameters. It does not perform training.

 python CNN.py

### Running `CNNMNIST.py`

This script downloads the MNIST dataset (if not already present), trains a custom ResNet model, and evaluates its performance. It includes data augmentation for training.

 python CNNMNIST.py

### Using `attention.py`

`attention.py` contains modular components for attention mechanisms. It's designed to be imported and used within larger models rather than run as a standalone script. There is no direct execution block (`if __name__ == '__main__':`) in this file.