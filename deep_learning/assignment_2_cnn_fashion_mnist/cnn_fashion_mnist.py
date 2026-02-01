########################################################################################################################
# Authors:     Amer Delic
# MatNr:       01331672
# File:        cnn_fashion_mnist.py
# Description: Deep Learning for AIE - Assignment 2
# Comments:    Convolutional Neural Networks
# Date: December 2025
# Requirements: numpy = 2.0.2/matplotlib = 3.9.4/torch = 2.8.0/torchvision = 0.23.0/pathlib = 1.0.1
########################################################################################################################
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from pathlib import Path

# ---------------------------------------- General settings ------------------------------------------------------------
# Ensure reproducibility
random.seed(99)
np.random.seed(99)
torch.manual_seed(99)
torch.cuda.manual_seed(99)
torch.cuda.manual_seed_all(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
gen = torch.Generator().manual_seed(99)

# Check GPU availability and ensure reproducibility if cuda is used
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#print(device)

# Setup path for model and figure saving
mdl_dir = Path('./models')
mdl_dir.mkdir(exist_ok=True)

fgr_dir = Path('./figures')
fgr_dir.mkdir(exist_ok=True)
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------- CNN model class -------------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, conv_layers: list, kernels_per_conv_layer: list, fc_layers: list, dropout_p: float = 0.0)-> None:
        super().__init__()

        num_in_channels = 1
        num_classes = 10
        layers_list = []

        # Convolutional and pooling layers
        for num_out_channels, kernel in zip(conv_layers, kernels_per_conv_layer):
            layers_list.append(nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels, kernel_size=kernel))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            num_in_channels = num_out_channels

        # Flattening the output of convolutional and pooling part
        layers_list.append(nn.Flatten())

        # Linear layers
        in_layer_size = fc_layers[0]
        for out_layer_size in fc_layers[1:]:
            layers_list.append(nn.Linear(in_features=in_layer_size, out_features=out_layer_size))
            layers_list.append(nn.ReLU())
            if dropout_p > 0.0:
                layers_list.append(nn.Dropout(p=dropout_p))
            in_layer_size = out_layer_size

        layers_list.append(nn.Linear(in_features=in_layer_size, out_features=num_classes))

        self.model = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.model(x)

    def get_param_num(self):
        param_num = 0
        for param in self.parameters():
            if param.requires_grad:
                param_num += param.numel()

        return param_num
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------- Training procedure -----------------------------------------------------------
def train_model(model, optimizer, train_loader, val_loader, device, num_epochs, patience, model_path,
                manual_l2_lambda=0.0, fig_save=False):
    
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define and initialize early stopping aux variables
    best_val_loss = np.inf
    best_train_loss = 0.0
    best_train_err = 0
    best_train_avg_correct = 0.0
    best_val_err = 0
    best_val_avg_correct = 0.0
    best_val_loss_epoch_nr = 0
    patience_cnt = 0

    # Training procedure including early stopping
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_avg_correct_epoch_list = []
    val_avg_correct_epoch_list = []
    for epoch in range(num_epochs):
        print("-"*20, f"Epoch {epoch}", "-"*20)

        # Train one epoch
        model.train()
        train_correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            train_loss = loss_fn(outputs, target)
            
            if manual_l2_lambda > 0.0:
                l2 = 0.0
                for p in model.parameters():
                    if p.requires_grad and p.dim() > 1:
                        l2 = l2 + (p ** 2).sum()
                train_loss = train_loss + manual_l2_lambda * l2
                
            train_loss.backward()
            optimizer.step()
            
            # Accuracy calculation of train set
            train_probs = F.softmax(outputs, dim=1)
            train_pred = torch.argmax(train_probs, dim=1)  # get the index of the max probability as the predicted output
            train_correct += (train_pred == target).sum().item()

            train_losses.append(train_loss.item())

        train_avg_correct = train_correct / len(train_loader.dataset)
        print(f"Average Training Loss {np.mean(train_losses[-len(train_loader):]):.4f}, "
              f"Accuracy: {train_correct}/{len(train_loader.dataset)}"
              f"({100. * train_avg_correct:.0f}%)")

        # Evaluate on validation set at the end of the epoch
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                val_loss += loss_fn(outputs, target).item()
                val_probs = F.softmax(outputs, dim=1)
                val_pred = torch.argmax(val_probs, dim=1)  # get the index of the max probability as the predicted output
                val_correct += (val_pred == target).sum().item()

        val_loss = val_loss / len(val_loader)
        val_avg_correct = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_avg_correct)
        print(f"Average Validation Loss: {val_loss:.4f}, Accuracy: {val_correct}/{len(val_loader.dataset)}"
              f"({100. * val_avg_correct:.0f}%)")
        
        if fig_save:
            train_avg_correct_epoch_list.append(train_avg_correct)
            val_avg_correct_epoch_list.append(val_avg_correct)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = np.mean(train_losses[-len(train_loader):])
            best_train_err = len(train_loader.dataset) - train_correct
            best_train_avg_correct = train_correct/len(train_loader.dataset)
            best_val_err = len(val_loader.dataset) - val_correct
            best_val_avg_correct = val_avg_correct
            best_val_loss_epoch_nr = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Stopping early at epoch {epoch+1}")
                print(f"Saving model from epoch {best_val_loss_epoch_nr}")
                break

    return (best_train_loss, best_val_loss, best_train_err, best_train_avg_correct,
            best_val_err, best_val_avg_correct,train_avg_correct_epoch_list, val_avg_correct_epoch_list)

def train_final_model(model, optimizer, train_loader, device, num_epochs):
    
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print("-"*20, f"Epoch {epoch}", "-"*20)

        model.train()
        batch_losses = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        print(f"Average Training Loss: {train_loss:.4f}")

    return train_loss

def evaluate_final_model(model, data_loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            total_loss += loss_fn(outputs, target).item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == target).sum().item()

    avg_loss = total_loss / len(data_loader)
    acc = correct / len(data_loader.dataset)
    err = 1.0 - acc
    return avg_loss, acc, err
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------- Task a) ---------------------------------------------------------------
# Define transform object
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load FashionMNIST dataset
full_train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)

# Split training dataset into new training and validation datasets
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [50000, 10000], generator=gen)

# Extract indices for later reuse
train_indices = train_dataset.indices
val_indices = val_dataset.indices

# Define batch size
BATCH_SIZE = 64

# Convert datasets to dataloaders
full_train_dataloader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=gen)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=gen)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=gen)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=gen)

# Define FashionMNIST categories/classes
classes = ("T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------- Task b) ---------------------------------------------------------------
# Define different network configurations
cnn_cfg = [
    {"conv_layers": [16], "kernels_per_conv_layer": [(3,3)], "fc_layers": [16*13*13, 120, 84], "lr": 1e-4},
    {"conv_layers": [16, 32], "kernels_per_conv_layer": [(3,3), (3,3)], "fc_layers": [32*5*5, 120, 84], "lr": 5e-4},
    {"conv_layers": [32, 64], "kernels_per_conv_layer": [(3,3), (3,3)], "fc_layers": [64*5*5, 120, 84], "lr": 3e-4},
    {"conv_layers": [16, 32, 64], "kernels_per_conv_layer": [(3,3), (3,3), (3,3)], "fc_layers": [64*1*1, 120, 84], "lr": 1e-3},
    {"conv_layers": [32, 64, 64], "kernels_per_conv_layer": [(5,5), (3,3), (3,3)], "fc_layers": [64*1*1, 120, 84], "lr": 1e-3}
    ]

# Train different different CNN networks
print("="*10, "Design Results", "="*10)

train_losses = []
val_losses = []
train_params = []
for cfg_idx, cfg_dict in enumerate(cnn_cfg):

    conv_layers = cfg_dict["conv_layers"]
    kernels_per_conv_layer = cfg_dict["kernels_per_conv_layer"]
    fc_layers = cfg_dict["fc_layers"]
    learning_rate = cfg_dict["lr"]

    # Create a CNN model
    cnn_model = ConvNet(conv_layers, kernels_per_conv_layer, fc_layers).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)

    # Define number of epochs and early stopping patience
    epochs = 100
    ptnc = 5

    # Define path for saving models
    mdl_path =  mdl_dir / f"model_cfg_{cfg_idx}.pth"

    print(f"CNN configuration:")
    print(f"Convolution layers: {str(conv_layers)} | Kernels per convolution layer: {str(kernels_per_conv_layer)} | "
          f"Fully connected layers: {str(fc_layers[1:])} | Learning rate: {learning_rate} | "
          f"Trainable parameters: {cnn_model.get_param_num()}")

    train_loss, val_loss, _, _, _, _, _, _ = train_model(cnn_model, optimizer, train_dataloader, val_dataloader, device,
                                       epochs, ptnc, mdl_path)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_params.append(cnn_model.get_param_num())


print("="*30)

for idx_loss, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
    print(f"Configuration {idx_loss+1} |"
          f"Convolution layers: {str(cnn_cfg[idx_loss]['conv_layers'])} | "
          f"Kernels per convolution layer: {str(cnn_cfg[idx_loss]['kernels_per_conv_layer'])} | "
          f"Fully connected layers: {str(cnn_cfg[idx_loss]['fc_layers'][1:])} | "
          f"Trainable parameters: {train_params[idx_loss]} | "
          f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------- Task c) ---------------------------------------------------------------
# Load selected model
mdl_idx = 2
cfg_dict = cnn_cfg[mdl_idx]
mdl_path =  mdl_dir / f"model_cfg_{mdl_idx}.pth"

conv_layers = cfg_dict["conv_layers"]
kernels_per_conv_layer = cfg_dict["kernels_per_conv_layer"]
fc_layers = cfg_dict["fc_layers"]

cnn_model_v2 = ConvNet(conv_layers, kernels_per_conv_layer, fc_layers)

cnn_model_v2.load_state_dict(torch.load(mdl_path))

# Extract kernels from the first convolutional layer
kernels_conv1 = cnn_model_v2.model[0].weight.detach()

# Kernels visualization
fig, axs = plt.subplots(4,4, figsize=(10, 8))
for i in range(16):
    row_idx = i // 4
    col_idx = i % 4
    axs[row_idx, col_idx].imshow(kernels_conv1[i][0], cmap='gray')
    axs[row_idx, col_idx].axis('off')
    axs[row_idx, col_idx].set_title(f"Kernel {i+1}", fontsize=10)
#plt.suptitle("Kernels visualization")
#plt.show()

# Save figure
fgr_pth = fgr_dir / f"kernels_visualization_cfg_{mdl_idx}.png"
fig.savefig(fgr_pth)
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------- Task d) ---------------------------------------------------------------
# --------------------------------- L2 Regularization investigation ----------------------------------------------------
print("="*10, "L2 Regularization Results", "="*10)

# Select best performing model configuration
conv_layers = cnn_cfg[mdl_idx]["conv_layers"]
kernels_per_conv_layer = cnn_cfg[mdl_idx]["kernels_per_conv_layer"]
fc_layers = cnn_cfg[mdl_idx]["fc_layers"]
learning_rate = cnn_cfg[mdl_idx]["lr"]

# Define different weight decays
wd_list = [{"manual_l2_lambda": 1e-3, "weight_decay": 0.0, "decoupled_weight_decay": False},
           {"manual_l2_lambda": 1e-4, "weight_decay": 0.0, "decoupled_weight_decay": False},
           {"manual_l2_lambda": 1e-5, "weight_decay": 0.0, "decoupled_weight_decay": False},
           {"manual_l2_lambda": 0.0, "weight_decay": 1e-3, "decoupled_weight_decay": False},
           {"manual_l2_lambda": 0.0, "weight_decay": 1e-4, "decoupled_weight_decay": False},
           {"manual_l2_lambda": 0.0, "weight_decay": 1e-5, "decoupled_weight_decay": False},
           {"manual_l2_lambda": 0.0, "weight_decay": 1e-3, "decoupled_weight_decay": True},
           {"manual_l2_lambda": 0.0, "weight_decay": 1e-4, "decoupled_weight_decay": True},
           {"manual_l2_lambda": 0.0, "weight_decay": 1e-5, "decoupled_weight_decay": True},
           ]

# Train CNN with different L2 regularization
train_losses = []
val_losses = []
train_avg_corrects = []
val_avg_corrects = []
for wd_idx, wd_dict in enumerate(wd_list):
    
    # Define weight decay
    lamda = wd_dict["manual_l2_lambda"]
    wd = wd_dict["weight_decay"]
    wd_decouple = wd_dict["decoupled_weight_decay"]

    # Create CNN model for regularization tests
    cnn_model_reg = ConvNet(conv_layers, kernels_per_conv_layer, fc_layers).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(cnn_model_reg.parameters(), lr=learning_rate, weight_decay=wd,
                                 decoupled_weight_decay=wd_decouple)

    # Define number of epochs and early stopping patience
    epochs = 100
    ptnc = 5

    # Define path for saving models
    mdl_path =  mdl_dir / f"model_cfg_l2_{wd_idx}.pth"

    print(f"L2 Regularization Investigation:")
    print(f"L2 lambda: {str(lamda)} | Weight decay: {str(wd)} | Decoupled weight decay: {str(wd_decouple)}")

    train_loss, val_loss, _, train_avg_correct, _, val_avg_correct, _, _  = train_model(cnn_model_reg,
                    optimizer, train_dataloader, val_dataloader, device, epochs, ptnc, mdl_path, manual_l2_lambda=lamda)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_avg_corrects.append(train_avg_correct)
    val_avg_corrects.append(val_avg_correct)
    

print("="*30)

for idx_loss, (train_loss, val_loss, train_avg_correct, val_avg_correct) in enumerate(
    zip(train_losses, val_losses, train_avg_corrects, val_avg_corrects)):
    print(f"Configuration {idx_loss+1} | L2 lambda: {wd_list[idx_loss]['manual_l2_lambda']} | "
        f"Weight decay: {wd_list[idx_loss]['weight_decay']} | "
        f"Decoupled weight decay: {wd_list[idx_loss]['decoupled_weight_decay']} | "
        f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | "
        f"Train Accuracy: {train_avg_correct*100:.2f}% | "
        f"Validation Accuracy: {val_avg_correct*100:.2f}% | "
        f"Train Error: {(1-train_avg_correct)*100:.2f}% | "
        f"Validation Errors: {(1-val_avg_correct)*100:.2f}%")
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- Droput investigation --------------------------------------------------------
print("="*10, "Dropout Results ", "="*10)
# Define different dropout rates
dropout_cfgs = [{"dropout_rate": 0.2, "fig_save": True},
               {"dropout_rate": 0.3, "fig_save": False},
               {"dropout_rate": 0.5, "fig_save": False}
              ]

# Train CNN with different dropout rates
train_losses = []
val_losses = []
train_avg_corrects = []
val_avg_corrects = []
train_avg_correct_list = []
val_avg_correct_list = []
for dropout_idx, dropout_cfg in enumerate(dropout_cfgs):

    dropout_rate = dropout_cfg["dropout_rate"]
    save_fig = dropout_cfg["fig_save"]
    
    # Create CNN model for regularization tests
    cnn_model_drop = ConvNet(conv_layers, kernels_per_conv_layer, fc_layers, dropout_rate).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(cnn_model_drop.parameters(), lr=learning_rate)

    # Define number of epochs and early stopping patience
    epochs = 100
    ptnc = 5

    # Define path for saving models
    mdl_path =  mdl_dir / f"model_cfg_drop_{dropout_idx}.pth"

    print(f"Dropout Investigation:")
    print(f"Convolution layers: {str(conv_layers)} | Kernels per convolution layer: {str(kernels_per_conv_layer)} | "
          f"Fully connected layers: {str(fc_layers[1:])} | Learning rate: {learning_rate} | "
          f"Dropout rate: {str(dropout_rate)} | "
          f"Trainable parameters: {cnn_model_drop.get_param_num()}")

    train_loss, val_loss, _, train_avg_correct, _, val_avg_correct, train_avg_correct_list, val_avg_correct_list = train_model(
        cnn_model_drop, optimizer, train_dataloader, val_dataloader, device, epochs, ptnc, mdl_path, fig_save=save_fig)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_avg_corrects.append(train_avg_correct)
    val_avg_corrects.append(val_avg_correct)


print("="*30)

for idx_loss, (train_loss, val_loss, train_avg_correct, val_avg_correct) in enumerate(
    zip(train_losses, val_losses, train_avg_corrects, val_avg_corrects)):
    print(f"Configuration {idx_loss+1} | Dropout rate: {str(dropout_cfgs[idx_loss]['dropout_rate'])} | "
          f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | "
          f"Train Accuracy: {train_avg_correct*100:.2f}% | "
          f"Validation Accuracy: {val_avg_correct*100:.2f}% | "
          f"Train Error: {(1-train_avg_correct)*100:.2f}% | "
          f"Validation Errors: {(1-val_avg_correct)*100:.2f}%")
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- Data Augmentation ----------------------------------------------------------
print("="*10, "Data Augmentation Results", "="*10)
# Define augmentation pipelines
transform_aug_1 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomCrop(28, padding=2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_aug_2 = transforms.Compose([
    transforms.RandomRotation(degrees=10),              
    transforms.ColorJitter(brightness=0.3, contrast=0.3), 
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),   
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_aug_3 = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
    transforms.RandomHorizontalFlip(p=0.5),               
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_aug_list = [transform_aug_1, transform_aug_2, transform_aug_3]

# Train CNN with different augmentation transformations
train_losses = []
val_losses = []
train_avg_corrects = []
val_avg_corrects = []

for transform_aug_idx, transform_aug in enumerate(transform_aug_list):
    
    # Load FashionMNIST dataset with augmentation
    full_train_dataset_aug = datasets.FashionMNIST('./data', train=True, download=True, transform=transform_aug)

    # Create train subset with the same elements as the original train subset without augmentation
    train_dataset_aug = torch.utils.data.Subset(full_train_dataset_aug, train_indices)

    # Convert datasets to dataloaders
    train_dataloader_aug = DataLoader(train_dataset_aug, batch_size=BATCH_SIZE, shuffle=True, generator=gen)

    # Create CNN model for regularization tests
    cnn_model_aug = ConvNet(conv_layers, kernels_per_conv_layer, fc_layers).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(cnn_model_aug.parameters(), lr=learning_rate)

    # Define number of epochs and early stopping patience
    epochs = 100
    ptnc = 5

    # Define path for saving models
    mdl_path =  mdl_dir / f"model_cfg_aug_{transform_aug_idx}.pth"

    print(f"Data Set Augmentation Investigation:")
    print(f"Convolution layers: {str(conv_layers)} | Kernels per convolution layer: {str(kernels_per_conv_layer)} | "
          f"Fully connected layers: {str(fc_layers[1:])} | Learning rate: {learning_rate} | "
          f"Trainable parameters: {cnn_model_aug.get_param_num()}")

    train_loss, val_loss, _, train_avg_correct, _, val_avg_correct, _, _ = train_model(cnn_model_aug, optimizer,
                                        train_dataloader, val_dataloader, device, epochs, ptnc, mdl_path)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_avg_corrects.append(train_avg_correct)
    val_avg_corrects.append(val_avg_correct)
    
print("="*30)

for idx_loss, (train_loss, val_loss, train_avg_correct, val_avg_correct) in enumerate(
    zip(train_losses, val_losses, train_avg_corrects, val_avg_corrects)):
    print(f"Transformation {idx_loss+1} | "
          f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | "
          f"Train Accuracy: {train_avg_correct*100:.2f}% | "
          f"Validation Accuracy: {val_avg_correct*100:.2f}% | "
          f"Train Error: {(1-train_avg_correct)*100:.2f}% | "
          f"Validation Errors: {(1-val_avg_correct)*100:.2f}%")
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Task e) ---------------------------------------------------------------
# Plot train and validation error during training for the best performing configuration
plt.figure(figsize=(10, 6))
plt.plot(list(range(1, len(train_avg_correct_list) + 1)), 1 - np.array(train_avg_correct_list), label="Train Error")
plt.plot(list(range(1, len(val_avg_correct_list) + 1)), 1 - np.array(val_avg_correct_list), label="Validation Error")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Evolution of training and validation errors during training")
plt.legend()
plt.grid(True)
#plt.show()
# Save figure
fgr_pth = fgr_dir / "Error_evolution.png"
plt.savefig(fgr_pth)

print("="*10, "Final Model Results", "="*10)
# Final model configuration
conv_layers = [32, 64]
kernels_per_conv_layer = [(3,3), (3,3)]
fc_layers = [64*5*5, 120, 84]
learning_rate = 3e-4
dropout_rate = 0.2

# Create a CNN model
cnn_model_final = ConvNet(conv_layers, kernels_per_conv_layer, fc_layers, dropout_rate).to(device)

# Define optimizer
optimizer = torch.optim.Adam(cnn_model_final.parameters(), lr=learning_rate)

# Define number of epochs
epochs = 20

print(f"Final CNN configuration:")
print(f"Convolution layers: {str(conv_layers)} | Kernels per convolution layer: {str(kernels_per_conv_layer)} | "
        f"Fully connected layers: {str(fc_layers[1:])} | Dropout rate: {str(dropout_rate)} | "
        f"Learning rate: {learning_rate} | Trainable parameters: {cnn_model_final.get_param_num()}")

_ = train_final_model(cnn_model_final, optimizer, full_train_dataloader, device, epochs)

final_test_loss, final_test_acc, final_test_err = evaluate_final_model(cnn_model_final, test_dataloader, device)
print("="*10, "Final Model Test Results", "="*10)
print(f"Test Loss: {final_test_loss:.4f}")
print(f"Test Accuracy: {100.0*final_test_acc:.2f}%")
print(f"Test Error: {100.0*final_test_err:.2f}%")
print("="*40)

# Confusion matrix
num_classes = 10
conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in test_dataloader:
        data, target = data.to(device), target.to(device)
        outputs = cnn_model_final(data)
        pred = torch.argmax(outputs, dim=1)

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

        for t, p in zip(target.view(-1), pred.view(-1)):
            conf_mat[t.long(), p.long()] += 1

conf_mat_np = conf_mat.numpy()

# Plot confusion matrix
plt.figure(figsize=(7, 6))
plt.imshow(conf_mat_np, cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.xticks(range(num_classes), classes, rotation=45, ha="right")
plt.yticks(range(num_classes), classes)
plt.tight_layout()
#plt.show()
# Save figure
fgr_pth = fgr_dir / "Confusion_matrix.png"
plt.savefig(fgr_pth)

# 5) Per-class accuracy
per_class_correct = np.diag(conf_mat_np)
per_class_total = conf_mat_np.sum(axis=1)
per_class_acc = per_class_correct / np.maximum(per_class_total, 1)

print("="*10, "Per-class accuracy (Test)", "="*10)
for i, cls in enumerate(classes):
    print(f"{i:2d} {cls:<12}: {100.0*per_class_acc[i]:6.2f}%  (correct {per_class_correct[i]}/{per_class_total[i]})")

# Showing a few most-confused pairs
conf_off = conf_mat_np.copy()
np.fill_diagonal(conf_off, 0)
flat_idx = np.argsort(conf_off.ravel())[::-1][:5]
print("\nMost confused pairs (top 5):")
for idx in flat_idx:
    r = idx // num_classes
    c = idx % num_classes
    if conf_off[r, c] > 0:
        print(f"True '{classes[r]}' predicted as '{classes[c]}' : {conf_off[r,c]} times")
# ----------------------------------------------------------------------------------------------------------------------