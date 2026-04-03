# Assignment 1 — Linear & Logistic Regression, Gradient Descent

## Overview
This assignment implements **linear regression**, **logistic regression**, and **gradient descent**  
on various datasets as part of the Machine Learning for AIE course (TU Graz, WS2025).

## Tasks Completed
- Implemented univariate linear regression with least-squares fitting and Pearson correlation analysis.
- Extended to **multiple linear regression** using the design matrix and pseudoinverse formulation.
- Applied **polynomial regression** to capture non-linear relationships between features.
- Trained **logistic regression classifiers** (via scikit-learn) on three datasets with different decision boundaries.
- Constructed appropriate design matrices including polynomial and interaction features.
- Evaluated models using accuracy and cross-entropy loss on train and test splits.
- Implemented **gradient descent with decaying learning rate** to minimize the Rastrigin function.
- Analyzed convergence behavior and visualized optimization trajectories.

## Files
- `main.py` — entry point to run all tasks
- `linear_regression.py` — implementation of univariate, multiple, and polynomial regression
- `logistic_regression.py` — logistic regression with design matrix construction and evaluation
- `gradient_descent.py` — gradient descent implementation on the Rastrigin function
- `plot_utils.py` — utility functions for plotting and visualization
- `requirements.txt` — required Python package versions
- `linear_logistic_regression_report.pdf` — report submitted for grading
