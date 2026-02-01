########################################################################################################################
# Authors:     Amer Delic
# MatNr:       01331672
# File:        nn_california_housing.py
# Description: Deep Learning for AIE - Assignment 1
# Comments:    Training Simple Neural Networks
# Date: November 2025
########################################################################################################################

import numpy as np 
import matplotlib.pyplot as plt
import torch

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

# ---------------------------------------- General settings ------------------------------------------------------------
# Ensure reproducibility
torch.manual_seed(99)
np.random.seed(99)

# Check GPU availability and ensure reproducibility if cuda is used
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(99)
    torch.cuda.manual_seed_all(99)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device("cpu")
#print(device)

# Fetch the dataset and split it in training and test datasets
X, y = fetch_california_housing(return_X_y=True)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.25, random_state=302)
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------- Task a) --------------------------------------------------------------- 
# Split training dataset in training and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=302)

# Normalize the dataset features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Plot feature distributions
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

plt.figure(figsize=(16, 12))
for i in range(len(feature_names)):
    plt.subplot(3, 3, i + 1)
    plt.hist(X_train[:, i], bins=50)
    plt.title(f'Distribution of {feature_names[i]}')
    plt.xlabel(feature_names[i])
    plt.ylabel('Count')
    plt.grid(False)

plt.tight_layout()
plt.show()
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------- Task b) ---------------------------------------------------------------
# Convert training dataset to a tensor dataset
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_train_ds = TensorDataset(X_train_tensor, y_train_tensor)

# Convert validation dataset to a tensor dataset
X_val_tensor = torch.from_numpy(X_val)
y_val_tensor = torch.from_numpy(y_val)
X_val_ds = TensorDataset(X_val_tensor, y_val_tensor)

# Define neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers: list, linear_layers=False):
        super().__init__()

        input_layer_size = 8
        output_layer_size = 1
        current_layer_size = input_layer_size
        layers_list = []

        for next_layer_size in hidden_layers:
            layers_list.append(nn.Linear(current_layer_size, next_layer_size))
            if not linear_layers:
                layers_list.append(nn.ReLU())
            current_layer_size = next_layer_size
        
        layers_list.append(nn.Linear(current_layer_size, output_layer_size))

        self.model = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.model(x)


# Define different configurations
configs = [
    {"hidden_layers": [32], "lr": 1e-3, "batch_size": 32, "epochs": 100, "save_loss_info": False},
    {"hidden_layers": [64], "lr": 1e-3, "batch_size": 32, "epochs": 100, "save_loss_info": False},
    {"hidden_layers": [128], "lr": 1e-3, "batch_size": 128, "epochs": 200, "save_loss_info": False},
    {"hidden_layers": [32,16], "lr": 1e-3, "batch_size": 32, "epochs": 100, "save_loss_info": False},
    {"hidden_layers": [64,32], "lr": 1e-3, "batch_size": 32, "epochs": 100, "save_loss_info": False},
    {"hidden_layers": [128,64], "lr": 1e-2, "batch_size": 64, "epochs": 100, "save_loss_info": False},
    {"hidden_layers": [64,32,16], "lr": 1e-3, "batch_size": 32, "epochs": 150, "save_loss_info": False},
    {"hidden_layers": [128,64,32], "lr": 1e-2, "batch_size": 64, "epochs": 100, "save_loss_info": True},
    {"hidden_layers": [64,64,64], "lr": 1e-3, "batch_size": 32, "epochs": 150, "save_loss_info": False},
    {"hidden_layers": [128,64,32,16], "lr": 1e-3, "batch_size": 32, "epochs": 200, "save_loss_info": False}
]

train_loss_avg_epoch_list = [] # list to save train losses of the final model
val_loss_avg_epoch_list = [] # list to save validation losses of the final model

# Train and compare different models
print("===== Design results =====")
for cfg in configs:

    # Get configuration data
    lr = cfg["lr"]
    bs = cfg["batch_size"]
    num_epochs = cfg["epochs"]
    layers_cfg = cfg["hidden_layers"]
    flg_save_loss = cfg["save_loss_info"]

    # Define Dataloaders based on configuration batch size
    X_train_loader = DataLoader(X_train_ds, batch_size=bs, shuffle=True)
    X_val_loader = DataLoader(X_val_ds, batch_size=bs, shuffle=True)

    # Create a model with a specific config
    model = NeuralNetwork(layers_cfg).to(device)

    # Define SGD optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fcn = nn.MSELoss()

    # Train current model
    for epoch in range(num_epochs):
        train_loss_epoch = 0.0
        train_samples_epoch = 0
        # Train one epoch
        model.train()
        for batch_data, batch_target in X_train_loader:
            batch_data, batch_target = batch_data.to(device).float(), batch_target.to(device).float()
            batch_preds = model(batch_data).squeeze()

            optimizer.zero_grad()
            loss = loss_fcn(batch_preds, batch_target)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item() * batch_data.size(0)
            train_samples_epoch += batch_data.size(0)
            
        train_loss_avg_epoch = train_loss_epoch/train_samples_epoch
        if flg_save_loss:
            train_loss_avg_epoch_list.append(train_loss_avg_epoch)


        # Evaluate on validation dataset
        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for batch_data, batch_target in X_val_loader:
                batch_data, batch_target = batch_data.to(device).float(), batch_target.to(device).float()
                batch_preds = model(batch_data).squeeze()

                loss = loss_fcn(batch_preds, batch_target)

                val_loss += loss.item() * batch_data.size(0)
                val_samples += batch_data.size(0)
            
        val_loss_avg_epoch = val_loss/val_samples
        if flg_save_loss:
            val_loss_avg_epoch_list.append(val_loss_avg_epoch)


    print(f"Hidden layers: {len(layers_cfg):<3} | Neurons per layers: {str(layers_cfg):<3} | Batch size: {bs:<3} |"
          f"Learning rate: {lr:<3} | Epochs: {num_epochs:<3} | Train loss: {train_loss_avg_epoch:.3f} |"
          f"Validation loss: {val_loss_avg_epoch:.3f}")

# ---------------------------------------------- Task c) ---------------------------------------------------------------
# Plot evolution of training and validation losses during training of the final model
plt.figure(figsize=(10, 6))
plt.plot(list(range(1, len(train_loss_avg_epoch_list) + 1)), train_loss_avg_epoch_list, label="Train Loss")
plt.plot(list(range(1, len(val_loss_avg_epoch_list) + 1)), val_loss_avg_epoch_list, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Evolution of training and validation losses during training")
plt.legend()
plt.grid(True)
plt.show()

# Define hyperparameters and hidden layers architecture of the final model
bs = 64
lr = 1e-2
num_epochs = 100
layers_cfg = [128,64,32]

# Normalize the dataset features
#scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

# Convert training and test datasets to tensor datasets and create dataloaders
X_train_full_tensor = torch.from_numpy(X_train_full)
y_train_full_tensor = torch.from_numpy(y_train_full)
X_train_full_ds = TensorDataset(X_train_full_tensor, y_train_full_tensor)
X_train_full_loader = DataLoader(X_train_full_ds, batch_size=bs, shuffle=True)

X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)
X_test_ds = TensorDataset(X_test_tensor, y_test_tensor)
X_test_loader = DataLoader(X_test_ds, batch_size=bs, shuffle=True)

# Create a model with a specific config
model = NeuralNetwork(layers_cfg).to(device)

# Define SGD optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_fcn = nn.MSELoss()

# Train the final model on the whole training set
for epoch in range(num_epochs):
    train_loss_epoch = 0.0
    train_samples_epoch = 0
    # Train one epoch
    model.train()
    for batch_data, batch_target in X_train_full_loader:
        batch_data, batch_target = batch_data.to(device).float(), batch_target.to(device).float()
        batch_preds = model(batch_data).squeeze()

        optimizer.zero_grad()
        loss = loss_fcn(batch_preds, batch_target)
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item() * batch_data.size(0)
        train_samples_epoch += batch_data.size(0)
        
    train_loss_avg_epoch = train_loss_epoch/train_samples_epoch


# Evaluate on test dataset
model.eval()
test_loss = 0
test_samples = 0
y_preds = []
y_true = []
with torch.no_grad():
    for batch_data, batch_target in X_test_loader:
        batch_data, batch_target = batch_data.to(device).float(), batch_target.to(device).float()
        batch_preds = model(batch_data).squeeze()

        y_preds.append(batch_preds)
        y_true.append(batch_target)

        loss = loss_fcn(batch_preds, batch_target)

        test_loss += loss.item() * batch_data.size(0)
        test_samples += batch_data.size(0)
    
test_loss_avg_epoch = test_loss/test_samples


print("===== Final model results =====")
print(f"Average training loss: {train_loss_avg_epoch:.3f} | Average test loss: {test_loss_avg_epoch:.3f}")


# Scatter plot - prediction vs ground truth
y_preds = torch.cat(y_preds).squeeze().numpy()
y_true = torch.cat(y_true).squeeze().numpy()

plt.scatter(y_preds, y_true, alpha=0.5)
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.title("Predictions vs Ground Truth")
plt.grid(True)
plt.show()

# ---------------------------------------------- Task d) ---------------------------------------------------------------
# Binary classification: is the house price below or above 200,000 USD

y_train_bin = y_train_full.copy()
y_test_bin = y_test.copy()

# Converting training targets in 0 or 1
for i in range(len(y_train_bin)):
    if y_train_bin[i] < 2.0:
        y_train_bin[i] = 0.0
    else:
        y_train_bin[i] = 1.0

# Converting test targets in 0 or 1
for i in range(len(y_test_bin)):
    if y_test_bin[i] < 2.0:
        y_test_bin[i] = 0.0
    else:
        y_test_bin[i] = 1.0

X_train_full_tensor_cls = torch.from_numpy(X_train_full).float()
y_train_full_tensor_cls = torch.from_numpy(y_train_bin).float()

X_test_tensor_cls = torch.from_numpy(X_test).float()
y_test_tensor_cls = torch.from_numpy(y_test_bin).float()

train_full_ds_cls = TensorDataset(X_train_full_tensor_cls, y_train_full_tensor_cls)
test_ds_cls = TensorDataset(X_test_tensor_cls, y_test_tensor_cls)

bs_cls = 64  # same batch size as in final regreession model
train_full_loader_cls = DataLoader(train_full_ds_cls, batch_size=bs_cls, shuffle=True)
test_loader_cls = DataLoader(test_ds_cls, batch_size=bs_cls, shuffle=False)

# Same architecture as in the final regression model
layers_cfg_cls = [128, 64, 32]
model_cls = NeuralNetwork(layers_cfg_cls).to(device)

lr_cls = 1e-3
num_epochs_cls = 100
optimizer_cls = torch.optim.SGD(model_cls.parameters(), lr=lr_cls)

def binary_cross_entropy(probs, targets):
    eps = 1e-8
    probs = torch.clamp(probs, eps, 1 - eps)
    loss = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
    return loss.mean()

print("===== Training binary classification model =====")
for epoch in range(num_epochs_cls):
    model_cls.train()
    train_loss_epoch = 0.0
    train_samples_epoch = 0

    for batch_data, batch_target in train_full_loader_cls:
        batch_data = batch_data.to(device).float()
        batch_target = batch_target.to(device).float()

        logits = model_cls(batch_data).squeeze()
        probs = torch.sigmoid(logits)

        optimizer_cls.zero_grad()
        loss = binary_cross_entropy(probs, batch_target)
        loss.backward()
        optimizer_cls.step()

        train_loss_epoch += loss.item() * batch_data.size(0)
        train_samples_epoch += batch_data.size(0)

    avg_train_loss_cls = train_loss_epoch / train_samples_epoch

    print(f"Epoch {epoch+1}/{num_epochs_cls} | Train BCE loss: {avg_train_loss_cls:.4f}")

# Evaluating the classification model on the test dataset
model_cls.eval()
test_loss_cls = 0.0
test_samples_cls = 0
correct_cls = 0

with torch.no_grad():
    for batch_data, batch_target in test_loader_cls:
        batch_data = batch_data.to(device).float()
        batch_target = batch_target.to(device).float()

        logits = model_cls(batch_data).squeeze()
        probs = torch.sigmoid(logits)

        loss = binary_cross_entropy(probs, batch_target)
        test_loss_cls += loss.item() * batch_data.size(0)
        test_samples_cls += batch_data.size(0)

        preds = (probs >= 0.5).float()

        correct_cls += (preds == batch_target).sum().item()

avg_test_loss_cls = test_loss_cls / test_samples_cls
test_accuracy_cls = correct_cls / test_samples_cls

print("===== Binary classification results =====")
print(f"Average test BCE loss: {avg_test_loss_cls:.4f}")
print(f"Test accuracy: {test_accuracy_cls:.4f}")

plt.figure(figsize=(8,6))
plt.hist(probs[batch_target == 0].cpu().numpy(), bins=30, alpha=0.5, label='Class 0')
plt.hist(probs[batch_target == 1].cpu().numpy(), bins=30, alpha=0.5, label='Class 1')
plt.title("Prediction probability distribution")
plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.legend()
plt.show()


# ----------------------------------------------------------------------------------------------------------------------