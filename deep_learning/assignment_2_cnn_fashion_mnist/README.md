# Assignment 2 — Convolutional Neural Network (Fashion-MNIST)

## Overview
This assignment focuses on building and training a **custom CNN** for multi-class image classification on the Fashion-MNIST dataset.

## Tasks Completed
- Loaded and prepared the dataset, including a validation split.
- Designed and trained multiple CNN architectures using Adam.
- Used early stopping to prevent overfitting.
- Compared at least 5 architectures with different depths and widths.
- Counted parameters manually for the best model and analyzed which layers dominate.
- Visualized first-layer convolutional kernels.
- Investigated regularization techniques:
  - L2 weight decay
  - Dropout
  - Data augmentation
- Summarized the final model architecture and training setup.
- Trained the final model on the full training set and reported:
  - Test accuracy  
  - Confusion matrix  
  - Training/validation error curves  

## Files
- `cnn_fashion_mnist.py` — full implementation of the CNN, training loop, evaluation, and visualizations
- `cnn_fashion_mnist_report.pdf` — report submitted for grading
