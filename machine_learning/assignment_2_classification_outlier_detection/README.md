# Assignment 2 — Course Project (Classification & Outlier Detection)

## Overview
This assignment is a **course project** on multi-class classification and outlier detection  
on a real-world sensor dataset as part of the Machine Learning for AIE course (TU Graz).  
The goal is to predict a discrete event y ∈ {0, 1, 2, 3} from 12 auxiliary sensor features,  
while also identifying anomalous (outlier) recordings using unsupervised detection methods.

## Tasks Completed
- Performed **Exploratory Data Analysis** including class distribution, feature boxplots, correlation heatmap, PCA projections, and inlier vs. outlier feature distributions.
- Designed and applied a **Stratified K-Fold cross-validation** strategy to handle class imbalance and avoid label leakage.
- Trained a **KNN baseline classifier** (K=9), evaluated using macro-averaged F1 score.
- Conducted experiments across 6 model families with varying hyperparameter configurations:
  - Logistic Regression, SVM (RBF kernel), Random Forest, Gradient Boosting, XGBoost, MLP
- Selected **Random Forest** as the best model based on macro F1 score.
- Implemented and compared three **outlier detection methods**: Gaussian Mixture Model, Local Outlier Factor, and Isolation Forest.
- Retrained the best classifier on GMM-filtered inlier data, achieving significant performance improvement.
- Generated predictions for both public leaderboard and final test sets.

## Files
- `main.py` — full implementation of EDA, model training, evaluation and outlier detection
- `requirements.txt` — required Python package versions
- `D.csv` — training dataset
- `D_out.csv` — known outlier samples for threshold calibration
- `D_test_leaderboard.csv` — public leaderboard test set
- `D_test_final.csv` — final leaderboard test set
- `classification_outlier_detection_report.pdf` — report submitted for grading
