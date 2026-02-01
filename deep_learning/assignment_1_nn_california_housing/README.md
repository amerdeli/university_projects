# Assignment 1 — Fully Connected Neural Network (California Housing)

## Overview
This assignment implements a **feedforward neural network** for a regression task on the California Housing dataset.  
The goal is to predict median house values using 8 numerical input features.

## Tasks Completed
- Split the dataset into training, validation, and test sets.
- Performed feature analysis and normalization using `StandardScaler`.
- Designed and trained multiple neural network architectures using SGD.
- Compared at least 10 configurations (hidden layers, units, learning rates, epochs).
- Selected the best model based on validation loss.
- Retrained the final model on the combined training + validation set.
- Evaluated performance on the test set and produced:
  - Training/validation loss curves  
  - Scatter plot of predictions vs. ground truth  
- Extended the model to a **binary classification** task (value < $200k vs. ≥ $200k).

## Files
- `nn_california_housing.py` — full implementation of preprocessing, model training, evaluation
- `nn_california_housing_report.pdf` — report submitted for grading
