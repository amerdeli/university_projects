########################################################################################################################
# Authors:     Amer Delic
# MatNr:       01331672
# File:        main.py
# Description: Machine Learning for AIE - Assignment 2
# Comments:    Course Project
# Date: January 2026
########################################################################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from xgboost import XGBClassifier
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_CLS_SCALER = None
_CLS_MODEL = None
_OUT_MODEL = None
_OUT_TAU = None

def predict(X_test):
    # TODO replace this with your model's predictions
    # For now, we will just return random predictions
    global _CLS_SCALER, _CLS_MODEL, _OUT_MODEL, _OUT_TAU

    if _CLS_MODEL is None or _CLS_SCALER is None or _OUT_MODEL is None or _OUT_TAU is None:
        labels = np.random.randint(4, size=len(X_test))
        outliers = np.random.randint(2, size=len(X_test))
        return labels, outliers

    X_feat = np.array(X_test.drop(columns=["id"]))

    X_cls = _CLS_SCALER.transform(X_feat)
    labels = _CLS_MODEL.predict(X_cls)

    if hasattr(_OUT_MODEL, "score_samples"):
        scores = _OUT_MODEL.score_samples(X_cls)
    elif hasattr(_OUT_MODEL, "decision_function"):
        scores = _OUT_MODEL.decision_function(X_cls)
    else:
        raise ValueError("Outlier model does not support scoring.")
    outliers = (scores < _OUT_TAU).astype(int)

    return labels, outliers

def generate_submission(test_data):
    label_predictions, outlier_predictions = predict(test_data)
    
    # IMPORTANT: stick to this format for the submission, 
    # otherwise your submission will result in an error
    submission_df = pd.DataFrame({ 
        "id": test_data["id"],
        "label": label_predictions,
        "outlier": outlier_predictions
    })
    return submission_df

def print_data_info(data: pd.DataFrame, data_outlier: pd.DataFrame) -> None:
    print("D shape:", data.shape)
    print("D_out shape:", data_outlier.shape)
    print("\nD info:")
    print(data.info())
    print("\nClass distribution in D:")
    print(data['label'].value_counts())

def create_eda_plots(X: pd.DataFrame, X_out: pd.DataFrame, X_all: pd.DataFrame, y: pd.Series) -> None:
    """
    Generates a full set of EDA plots:
    1. Distribution of class labels
    2. Feature correlation heatmap
    3. Feature boxplots
    4. PCA projection (colored by class)
    5. PCA projection (inliers vs outliers)
    6. Feature histograms (inliers vs outliers)
    """
    # -------------------------------------------
    # 1. Distribution of class labels
    # -------------------------------------------
    class_label_counts = y.value_counts().sort_index()
    plt.figure(figsize=(8,6))
    plt.bar(np.array(class_label_counts.index), np.array(class_label_counts.values))
    plt.xticks(ticks=np.array(class_label_counts.index), labels=[str(x) for x in class_label_counts.index])
    plt.title("Distribution of class labels")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.tight_layout()
    #plt.show()

    # -------------------------------------------
    # 2. Feature correlation heatmap
    # -------------------------------------------
    C = X.corr()
    feature_names = [f"x$_{{{i}}}$" for i in range(1,13)]
    #C.columns = feature_names
    #C.index = feature_names

    plt.figure(figsize=(8,6))
    sns.heatmap(C, cmap='coolwarm', square=True)
    plt.title("Feature correlation heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    #plt.show()

    # -------------------------------------------
    # 3. Feature boxplots
    # -------------------------------------------
    plt.figure(figsize=(8,6))
    plt.boxplot(X.values, tick_labels=feature_names)
    plt.title("Feature boxplots")
    plt.xlabel("Features")
    plt.ylabel("Value range")
    plt.tight_layout()
    #plt.show()

    # -------------------------------------------
    # 4. PCA projection (colored by class)
    # -------------------------------------------
    # Normalize dataset features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca_model = PCA(n_components=2)
    Z = pca_model.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))
    scatter_ax = plt.scatter(Z[:,0], Z[:,1], c=y, cmap='viridis', alpha=0.8)
    plt.xlabel(f"z$_{1}$")
    plt.ylabel(f"z$_{2}$")
    plt.title("PCA projection")
    plt.legend(*scatter_ax.legend_elements(), title="Class", loc="best")
    plt.tight_layout()
    #plt.show()

    # -------------------------------------------
    # 5. PCA projection (inliers vs outliers)
    # -------------------------------------------
    # Normalize dataset features including outliers
    scaler_all = StandardScaler()
    X_all_scaled = scaler_all.fit_transform(X_all)

    # Apply PCA
    pca_model_all = PCA(n_components=2)
    Z_all = pca_model_all.fit_transform(X_all_scaled)

    # Inlier/outlier mask
    inlier_mask = np.array([True]*len(X) + [False]*len(X_out))
    outlier_mask = ~inlier_mask

    plt.figure(figsize=(8,6))
    plt.scatter(Z_all[inlier_mask, 0], Z_all[inlier_mask, 1], c='blue', alpha=0.8, label='Inliers')
    plt.scatter(Z_all[outlier_mask, 0], Z_all[outlier_mask, 1], c='red', alpha=0.8, label='Outliers')
    plt.xlabel(f"z$_{1}$")
    plt.ylabel(f"z$_{2}$")
    plt.title("PCA projection - Inlier vs Outlier")
    plt.legend(loc="best")
    plt.tight_layout()
    #plt.show()

    # -------------------------------------------
    # 6. Feature histograms (inliers vs outliers)
    # -------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(22, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(X.columns):
        ax = axes[idx]
        
        sns.kdeplot(data=X, x=feature, ax=ax, label="Inliers", fill=True, alpha=0.5, color="blue")
        sns.kdeplot(data=X_out, x=feature, ax=ax, label="Outliers", fill=True, alpha=0.5, color="red")

        ax.set_xlabel(feature)
        ax.set_ylabel("Distribution")
        ax.legend()

    fig.suptitle("Feature distribution - Inlier vs Outlier", fontsize = 20)
    plt.tight_layout()
    plt.show()

def binary_accuracy_per_class(y_true: np.ndarray, y_pred: np.ndarray, classes: list) -> list:
    
    acc_per_class = []
    for cls in classes:
         
        # One-vs-rest binary
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        
        # Binary accuracy
        acc_per_class.append(accuracy_score(y_true_bin, y_pred_bin))
    
    return acc_per_class

def train_model(model, X: np.ndarray, y: np.ndarray, skf: StratifiedKFold) -> dict:
    """
    Trains the given model. It applies 10-fold stratified cross-validation on a standardized feature set and computes
    the per-class metrics and macro-averaged F1 score.
    The performance metrics are returen as a dictionary.
    """
    # Define output dict
    results = {}

    # Performance metrics storage lists
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_accuracies = []
    accuracies = []
    macro_f1_scores = []

    # Train the model
    for train_idx, val_idx in skf.split(X, y):
        
        # Split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize per fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Clone a new model instance for each fold
        model= clone(model)
        
        # Model training
        model.fit(X=X_train, y=y_train)
        y_pred = model.predict(X=X_val)

        # Per-class metrics
        per_class_precision.append(precision_score(y_val, y_pred, average=None, zero_division=0))
        per_class_recall.append(recall_score(y_val, y_pred, average=None, zero_division=0))
        per_class_f1.append(f1_score(y_val, y_pred, average=None, zero_division=0))
        per_class_accuracies.append(binary_accuracy_per_class(y_val, y_pred, [0, 1, 2, 3]))
            
        # Accuracy
        accuracies.append(accuracy_score(y_val, y_pred))
            
        # Macro F1 for model selection
        macro_f1_scores.append(f1_score(y_val, y_pred, average='macro'))
        
    # Store aggregated results
    results = { "precision_per_class": np.mean(per_class_precision, axis=0),
                "recall_per_class": np.mean(per_class_recall, axis=0),
                "f1_per_class": np.mean(per_class_f1, axis=0),
                "accuracy_per_class": np.mean(per_class_accuracies, axis=0),
                "accuracy": np.mean(accuracies),
                "macro_f1": np.mean(macro_f1_scores)}
    
    return results

def print_result_comparison(results: dict) -> None:
    
    print(f"{'Config ID':<3} | {'Precision per class':<29} | {'Recall per class':<29} | {'F1 score per class':<29}"
          f" | {'Accuracy per class':<29} | {'Accuracy':<5} | {'Macro F1 score'}")
    print(165*"-")

    for idx, result in results.items():
        precision = np.round(result["precision_per_class"], 4)
        recall = np.round(result["recall_per_class"], 4)
        f1 = np.round(result["f1_per_class"], 4)
        acc_binary = np.round(result["accuracy_per_class"], 4)
        acc = np.round(result["accuracy"], 4)
        macro_f1 = np.round(result["macro_f1"], 4)
        print(f"{idx+1:<9} | {precision} | {recall} | {f1} | {acc_binary} | {acc:<8} | {macro_f1}")

class MLP(nn.Module):
    def __init__(self, hidden_layers: list, dropout_p: float = 0.0):
        super().__init__()

        input_layer_size = 12
        output_layer_size = 4
        current_layer_size = input_layer_size
        layers_list = []

        for next_layer_size in hidden_layers:
            layers_list.append(nn.Linear(current_layer_size, next_layer_size))
            layers_list.append(nn.ReLU())
            if dropout_p > 0.0:
                layers_list.append(nn.Dropout(p=dropout_p))
            current_layer_size = next_layer_size
        
        layers_list.append(nn.Linear(current_layer_size, output_layer_size))

        self.model = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.model(x)

def train_mlp(mlp_model_layers: list, X: np.ndarray, y: np.ndarray, skf: StratifiedKFold,
              epochs=200, lr=1e-3, bs=32, patience=10, mlp_model_dropout=0.0) -> dict:
    
    
    """ Trains an MLP using StratifiedKFold cross-validation. 
        Returns the same metrics as the original train_model function.
    """
    # Define output dict
    results = {}
    
    # Performance metrics storage lists
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_accuracies = []
    accuracies = []
    macro_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        
        #print("="*20, f"SKF Fold {fold+1}", "="*20)
        
        # Split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize per fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Convert to tensors and dataloaders
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        #val_ds = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        #val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
        
        # Clone a new model instance for each fold
        mlp_model = MLP(mlp_model_layers, mlp_model_dropout)
        
        # Define optimizer and loss
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss() 
        
        # Training loop
        best_val_loss = np.inf
        patience_cnt = 0
        for epoch in range(epochs):
            #print("-"*20, f"Epoch {epoch}", "-"*20)
            mlp_model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad() 
                logits = mlp_model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
            # Validation (collect predictions)
            mlp_model.eval()
            val_correct = 0
            with torch.no_grad():
                logits = mlp_model(X_val_t)
                val_loss = criterion(logits, y_val_t).item()
                y_probs = F.softmax(logits, dim=1)
                y_pred = torch.argmax(y_probs, dim=1).numpy()
                val_correct += (y_pred == y_val_t).sum().item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_y_pred = y_pred
                best_epoch = epoch
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    #print(f"Stopping early at epoch {epoch+1}. Best epoch: {best_epoch}")
                    break
            
            #print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_correct}/{len(y_val_t)}")
            
        # Metrics
        per_class_precision.append(precision_score(y_val, best_y_pred, average=None, zero_division=0))
        per_class_recall.append(recall_score(y_val, best_y_pred, average=None, zero_division=0))
        per_class_f1.append(f1_score(y_val, best_y_pred, average=None, zero_division=0))
        
        # Your custom accuracy per class
        per_class_accuracies.append(binary_accuracy_per_class(y_val, best_y_pred, [0, 1, 2, 3]))
        accuracies.append(accuracy_score(y_val, best_y_pred))
        macro_f1_scores.append(f1_score(y_val, best_y_pred, average='macro'))
        
    # Aggregate results
    results = { "precision_per_class": np.mean(per_class_precision, axis=0),
                   "recall_per_class": np.mean(per_class_recall, axis=0),
                   "f1_per_class": np.mean(per_class_f1, axis=0), 
                   "accuracy_per_class": np.mean(per_class_accuracies, axis=0),
                   "accuracy": np.mean(accuracies),
                   "macro_f1": np.mean(macro_f1_scores)
                   } 
        
    return results

def select_gmm_components(X: np.ndarray, max_components=10, print_results=True) -> int: 

    """
    Fit multiple Gaussian Mixture Models with varying numbers of components 
    and evaluate their Bayesian Information Criterion (BIC) scores.
    """

    # Empty list for BIC scores 
    bic_scores = []
    
    # Fit different GMM models and evaluate their BIC scores
    for k in range(1, max_components+1):
        gmm_model = GaussianMixture(n_components=k, covariance_type='full', random_state=99)
        gmm_model.fit(X)
        bic_scores.append(np.round(gmm_model.bic(X),4))
        best_k = np.argmin(bic_scores) + 1

    # Print results
    if print_results:
        print(f"{'Components':<3} | {'BIC score':<29}")
        print(24*"-")
        for idx, bic_score in enumerate(bic_scores):
            print(f"{idx+1:<10} | {bic_score}")

    return int(best_k)

def fit_gmm_and_threshold(X: np.ndarray, X_out: np.ndarray, n_components: int) -> tuple:

    """
    Train a Gaussian Mixture Model on inlier data and determine a threshold 
    for outlier detection using an outlier dataset.
    """

    # Create and fit a GMM
    gmm_model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=99)
    gmm_model.fit(X)

    # Log probabilites for outlier dataset
    log_probs_out = gmm_model.score_samples(X_out)

    # Threshold (higher likelihood lower probability)
    threshold = np.max(log_probs_out)
    
    return gmm_model, float(threshold)

def filter_dataset(X: np.ndarray, y: np.ndarray, model, threshold: float) -> tuple:
     
    """
    Filter a dataset by removing samples classified as outliers based on 
    log-likelihood scores from a trained outlier detection model.
    """
        
    log_probs = model.score_samples(X)
    mask = log_probs > threshold
    return X[mask], y[mask]

def select_best_config(results: dict) -> int:
    best_idx = 0
    best_score = -np.inf
    for idx, res in results.items():
        if res["macro_f1"] > best_score:
            best_score = res["macro_f1"]
            best_idx = idx
    return best_idx

def fit_iso_forest_and_threshold(X: np.ndarray, X_out: np.ndarray, n_estimators = 400) -> tuple:

    """
    Trains IsolationForest on X and calibrates a threshold using X_out
    """

    iso_forest_model = IsolationForest(n_estimators=n_estimators,
                                       contamination='auto',
                                       n_jobs=-1,
                                       random_state=99
                                       )
    iso_forest_model.fit(X)

    # scores_in = iso_forest_model.decision_function(X)
    # scores_out = iso_forest_model.decision_function(X_out)
    scores_in = iso_forest_model.score_samples(X)
    scores_out = iso_forest_model.score_samples(X_out)

    taus = np.quantile(np.concatenate([scores_in, scores_out]), np.linspace(0.01, 0.50, 200))

    target_outlier_rate = 0.20
    best_tau = taus[0]
    best_obj = -np.inf

    for tau in taus:
        pred_out_on_out = (scores_out < tau).astype(int)
        pred_out_on_in = (scores_in < tau).astype(int)

        recall_out = np.mean(pred_out_on_out)
        outlier_rate_in = np.mean(pred_out_on_in)

        obj = 0.75 * recall_out - 0.25 * np.abs(outlier_rate_in - target_outlier_rate)
        if obj > best_obj:
            best_obj = obj
            best_tau = tau

    return iso_forest_model, best_tau

def fit_lof_and_threshold(X: np.ndarray, X_out: np.ndarray, n_neighbors: int = 20) -> tuple:
    
    """ 
    Train a Local Outlier Factor (LOF) model on inlier data and determine
    a threshold for outlier detection using an outlier dataset.
    """ 
    # Create and fit LOF (novelty=True allows scoring new samples)
    lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof_model.fit(X)
    
     # LOF scores for outlier dataset (more negative = more anomalous)
    lof_scores_out = lof_model.score_samples(X_out)
    
     # Threshold: highest LOF score among outliers
    threshold = np.max(lof_scores_out)
    
    return lof_model, float(threshold)

def evaluate_outlier_models_and_train_xgb(X: np.ndarray, y: np.ndarray, X_out: np.ndarray, skf,
                                            outlier_models: list = ['gmm','lof', 'iso_forest']) -> dict:  
    """
    Evaluate multiple outlier detection methods and train an XGBoost classifier 
    on the filtered datasets produced by each method.
    """
    # Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_out_scaled = scaler.transform(X_out)

    results = {}
    for idx, outlier_model in enumerate(outlier_models):

        if outlier_model == 'lof': 
            # Fit a LOF model and calibrate threshold
            model, outlier_threshold = fit_lof_and_threshold(X_scaled, X_out_scaled, n_neighbors=15)
        elif outlier_model == 'iso_forest':
            # Fit IsoForest model and calibrate threshold
            model, outlier_threshold = fit_iso_forest_and_threshold(X_scaled, X_out_scaled, n_estimators=400)
        elif outlier_model == 'gmm':
            # Determine optimal components number and fit a GMM and calibrate threshold
            gmm_components = select_gmm_components(X_scaled, print_results=False)
            model, outlier_threshold = fit_gmm_and_threshold(X_scaled, X_out_scaled, gmm_components)

        # Filter original dataset
        X_filt, y_filt = filter_dataset(X_scaled, y, model, outlier_threshold)

        # Temp scaler
        scaler_temp = StandardScaler()
        X_filt = scaler_temp.fit_transform(X_filt)

        # Retrain XGB model
        rf_cfg = {"n_estimators": 600, "max_depth": None, "max_features": "sqrt"}

        rf_model = RandomForestClassifier(n_estimators=rf_cfg["n_estimators"],
                                          max_depth=rf_cfg["max_depth"],
                                          max_features=rf_cfg["max_features"],
                                          n_jobs=-1,
                                          random_state=99)
    
        results[idx] = train_model(rf_model, X_filt, y_filt, skf)

    return results

def main():

#============================================ General part =============================================================
    # Ensure reproducibility
    torch.manual_seed(99)
    np.random.seed(99)
    
    # Read data 
    df_D = pd.read_csv("D.csv")
    df_D_out = pd.read_csv("D_out.csv")

    # Feature notation
    feature_names = [f"x$_{{{i}}}$" for i in range(1,13)]

    # Feature matrix 
    X = df_D.drop(columns=['id','label'])
    X.columns = feature_names
    
    # Feature matrix outliers
    X_out = df_D_out.drop(columns=['id'])
    X_out.columns = feature_names

    # Feature matrix (inliers and outliers)
    X_all = pd.concat([X, X_out], axis=0, ignore_index=True)
    X_all.columns = feature_names

    # Output vector
    y = df_D['label']

    # Normalize feature dataset
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # Validation strategy
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=99)
#=======================================================================================================================


#======================================= Exploratory Data Analysis =====================================================
    # # Create EDA plots
    create_eda_plots(X, X_out, X_all, y)

    # Convert to numpy arrays for further tasks
    X = np.array(X)
    y = np.array(y)
    X_out = np.array(X_out)
#=======================================================================================================================

#============================================ Baseline Model ===========================================================
    # List of different Ks
    k_list = [1, 3, 5, 7, 9, 11, 15, 21]

    # Compare different KNN models
    knn_results = {}
    for k_idx, k in enumerate(k_list):

        # Define the model
        knn_model = KNeighborsClassifier(n_neighbors=k)

        # Train the model
        knn_results[k_idx] = train_model(knn_model, X, y, skf)
    
    # Compare results
    print("Baseline Model Selection Results:")
    print_result_comparison(knn_results)
#=======================================================================================================================

#==================================== Model Experimentation and Validation =============================================
    
    # Logistic Regression
    logreg_configs = [
                {"C": 0.1, "penalty": "l2", "solver": "lbfgs"},
                {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
                {"C": 10.0, "penalty": "l2", "solver": "lbfgs"}
                ]

    logreg_results = {}
    for logreg_idx, logreg_config in enumerate(logreg_configs):

        logreg_model = LogisticRegression(C=logreg_config["C"],
                                          penalty=logreg_config["penalty"],
                                          solver=logreg_config["solver"],
                                          #multi_class="multinomial",
                                          max_iter=2000,
                                          random_state=99)

        logreg_results[logreg_idx] = train_model(logreg_model, X, y, skf)

    print("Logistic Regression Model Selection Results:")
    print_result_comparison(logreg_results)

    # Support Vector Machines
    svc_configs = [
                {"C": 1.0, "gamma": "scale", "kernel": "rbf"},
                {"C": 3.0, "gamma": "scale", "kernel": "rbf"},
                {"C": 10.0, "gamma": "scale", "kernel": "rbf"}
                ]

    svc_results = {}
    for svc_idx, svc_config in enumerate(svc_configs):

        svc_model = SVC(C=svc_config["C"],
                        gamma=svc_config["gamma"],
                        kernel=svc_config["kernel"],
                        random_state=99)

        svc_results[svc_idx] = train_model(svc_model, X, y, skf)

    print("SVM Model Selection Results:")
    print_result_comparison(svc_results)

    # Random Forest
    rf_configs = [
                {"n_estimators": 300, "max_depth": None, "max_features": "sqrt"},
                {"n_estimators": 600, "max_depth": None, "max_features": "sqrt"},
                {"n_estimators": 600, "max_depth": 12, "max_features": "sqrt"}
                ]

    rf_results = {}
    for rf_idx, rf_config in enumerate(rf_configs):

        rf_model = RandomForestClassifier(n_estimators=rf_config["n_estimators"],
                                          max_depth=rf_config["max_depth"],
                                          max_features=rf_config["max_features"],
                                          n_jobs=-1,
                                          random_state=99)

        rf_results[rf_idx] = train_model(rf_model, X, y, skf)

    print("Random Forest Model Selection Results:")
    print_result_comparison(rf_results)

    # Gradient Boosting Trees
    gboost_configs = [
                {"n_estimators": 100, "learning_rate": 0.2, "max_depth": 4, "subsample": 0.8, "max_features": None},
                {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.8, "max_features": None},
                {"n_estimators": 300, "learning_rate": 0.2, "max_depth": 3, "subsample": 1, "max_features": None}
                ]

    # Compare different Gradient Boost Tree models
    gboost_results = {}
    for gboost_idx, gboost_config in enumerate(gboost_configs):
    
        # Define the model 
        gboost_model = GradientBoostingClassifier(learning_rate=gboost_config["learning_rate"],
                                                n_estimators=gboost_config["n_estimators"],
                                                subsample=gboost_config["subsample"],
                                                max_depth=gboost_config["max_depth"],
                                                max_features=gboost_config["max_features"],
                                                random_state=99)
    
        # Train the model
        gboost_results[gboost_idx] = train_model(gboost_model, X, y, skf)
    
    # Compare results
    print("Gradient Boosting Model Selection Results:")
    print_result_comparison(gboost_results)
        
    # Gradient Boosting Trees - XGBoost
    xgboost_configs = [
                {"n_estimators": 500, "learning_rate": 0.1, "max_depth": 3, "reg_lambda": 1},
                {"n_estimators": 800, "learning_rate": 0.05, "max_depth": 5, "reg_lambda": 1},
                {"n_estimators": 800, "learning_rate": 0.1, "max_depth": 5, "reg_lambda": 3},
                {"n_estimators": 1200, "learning_rate": 0.05, "max_depth": 5, "reg_lambda": 3},
                {"n_estimators": 1200, "learning_rate": 0.1, "max_depth": 5, "reg_lambda": 5}
                ]

    # Compare different Gradient Boost Tree models
    xgboost_results = {}
    for xgboost_idx, xgboost_config in enumerate(xgboost_configs):
    
        # Define the model 
        xgboost_model = XGBClassifier(objective="multi:softprob",
                                      num_class=4,
                                      tree_method="hist",
                                      n_jobs=-1,
                                      learning_rate=xgboost_config["learning_rate"],
                                      n_estimators=xgboost_config["n_estimators"],
                                      max_depth=xgboost_config["max_depth"],
                                      reg_lambda=xgboost_config["reg_lambda"],
                                      subsample=0.8,
                                      random_state=99)

     
        # Train the model
        xgboost_results[xgboost_idx] = train_model(xgboost_model, X, y, skf)
    
    # Compare results
    print("XGBoost Model Selection Results:")
    print_result_comparison(xgboost_results)
    
    # MLP 
    mlp_configs = [
                # {"hidden_layers": [32,32], "dropout": 0.0},
                # {"hidden_layers": [32,64], "dropout": 0.0},
                {"hidden_layers": [64,64],  "dropout": 0.0},
                {"hidden_layers": [64,128], "dropout": 0.0},
                {"hidden_layers": [32,64,128], "dropout": 0.0},
                {"hidden_layers": [64,64], "dropout": 0.1},
                {"hidden_layers": [64,128], "dropout": 0.1},
                {"hidden_layers": [32,64,128], "dropout": 0.05}
                ]
    
    # Compare different MLP architecures
    mlp_results = {}
    for mlp_idx, mlp_config in enumerate(mlp_configs):
    
        # Train the model
        mlp_results[mlp_idx] = train_mlp(mlp_config["hidden_layers"], X, y, skf,
                                         mlp_model_dropout=mlp_config["dropout"])
        
    # Compare results
    print("MLP Model Selection Results:")
    print_result_comparison(mlp_results)
#=======================================================================================================================

#============================================ Outlier detection ========================================================
    
    # Compare different outlier models
    print("Outlier Detection Model Selection Results:")

    outlier_results = evaluate_outlier_models_and_train_xgb(X, y, X_out, skf)

    # Compare results
    print("Outlier Detection Model Results:")
    print_result_comparison(outlier_results)
#=======================================================================================================================

#============================================ Final configuration ======================================================
    
    # Scaler
    cls_scaler = StandardScaler()
    X = cls_scaler.fit_transform(X)
    X_out = cls_scaler.transform(X_out)

    # Fit a GMM and calibrate outlier threshold
    gmm_components = select_gmm_components(X, print_results=False)
    gmm_model, outlier_threshold = fit_gmm_and_threshold(X, X_out, gmm_components)

    # Filter original dataset
    X_filt, y_filt = filter_dataset(X, y, gmm_model, outlier_threshold)

    # Final model
    final_rf_cfg = {"n_estimators": 600, "max_depth": None, "max_features": "sqrt"}

    final_rf_model = RandomForestClassifier(n_estimators=final_rf_cfg["n_estimators"],
                                          max_depth=final_rf_cfg["max_depth"],
                                          max_features=final_rf_cfg["max_features"],
                                          n_jobs=-1,
                                          random_state=99)
    
    # Fit the final model
    final_rf_model.fit(X_filt, y_filt)
    
    global _CLS_SCALER, _CLS_MODEL, _OUT_MODEL, _OUT_TAU
    _CLS_SCALER = cls_scaler
    _CLS_MODEL = final_rf_model
    _OUT_MODEL = gmm_model
    _OUT_TAU = outlier_threshold
    
    df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
    submission_df = generate_submission(df_leaderboard)
    # IMPORTANT: The submission file must be named "submission_leaderboard_GroupName.csv",
    # replace GroupName with a group name of your choice. If you do not provide a group name, 
    # your submission will fail!
    submission_df.to_csv("submission_leaderboard_BraDe.csv", index=False)
    
    # For the final leaderboard, change the file name to "submission_final_GroupName.csv"
    df_final = pd.read_csv("D_test_final.csv")
    submission_df = generate_submission(df_final)
    submission_df.to_csv("submission_final_BraDe.csv", index=False)

    # Confusion matrix
    cm_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=99)
    cm_train_idx, cm_test_idx = next(cm_skf.split(X_filt, y_filt))
    X_cm_train, X_cm_test = X_filt[cm_train_idx], X_filt[cm_test_idx]
    y_cm_train, y_cm_test = y_filt[cm_train_idx], y_filt[cm_test_idx]
    cm_model = RandomForestClassifier(n_estimators=final_rf_cfg["n_estimators"],
                                        max_depth=final_rf_cfg["max_depth"],
                                        max_features=final_rf_cfg["max_features"],
                                        n_jobs=-1,
                                        random_state=99)
    cm_model.fit(X_cm_train, y_cm_train)
    y_cm_pred = cm_model.predict(X_cm_test)
    cm = confusion_matrix(y_cm_test, y_cm_pred, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax)
    ax.set_title("Confusion Matrix (Inliers Only) - Held-out Fold")
    fig.tight_layout()
    fig.savefig("confusion_matrix_inliers.png", dpi=200)
    plt.close(fig)
#=======================================================================================================================

if __name__ == "__main__":
    main()