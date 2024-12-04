import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LSTM import LSTM_Model

# the testing dataset
data_file_path = "./data/enstars/enstars_data_test2.csv"
data_file_columns = ["banner"]
data_names_path = "./data/enstars/enstars_names_test2.csv"

# the complete dataset of possible mappings of numbers to strings
mapping_file_path = "./data/enstars/enstars_data.csv"
mapping_file_columns = ["banner"]
mapping_names_path = "./data/enstars/enstars_names.csv"

# read csv
data_csv = pd.read_csv(data_file_path, usecols=data_file_columns)
names_csv = pd.read_csv(data_names_path, usecols=data_file_columns)

mapping_csv = pd.read_csv(mapping_file_path, usecols=mapping_file_columns)
mapping_names_csv = pd.read_csv(mapping_names_path, usecols=mapping_file_columns)

mapping = {}
for idx, name in zip(mapping_csv[mapping_file_columns[0]], mapping_names_csv[mapping_file_columns[0]]):
    mapping[idx] = name

# convert data to array
test_data_csv = data_csv.to_numpy().astype("float32")

# create tensors of the data and the window to lookback in for our LSTM
def create_dataset(data, window):
    features, targets = [], []
    target = []
    for i in range(len(data) - window):
        feat = data[i:i+window]
        target = data[i+1:i+window+1]
        features.append(feat)
        targets.append(target)

    features.append(target)
    targets.append(target)
    features_tensor = torch.tensor(features)
    targets_tensor = torch.tensor(targets)

    return features_tensor, targets_tensor

# manual normalization
def normalize(arr, mean, std):
    newArr = []
    for n in arr:
        newArr.append((n - mean) / std)
    return newArr

# manual denormalization 
def denormalize(arr, mean, std):
    newArr = []
    for n in arr:
        newArr.append((n * std) + mean)
    return newArr

def convertToString(arr):
    strings = []
    for n in arr:
        strings.append(mapping[round(n.item())])
    return strings

test_mean, test_std = test_data_csv.mean().item(), test_data_csv.std().item()

test_data_csv = normalize(test_data_csv, test_mean, test_std)

test_data_features, test_data_targets = create_dataset(test_data_csv, window=2)

# load data
test_dataloader = DataLoader(TensorDataset(test_data_features, test_data_targets), shuffle=False, batch_size=16)

# define constants
input_size = 1
hidden_size = 37
output_size = 1
num_epochs = 350
num_layers = 2
batch_first = True

model = LSTM_Model(input_size, hidden_size, num_layers, batch_first, output_size)
model.load_state_dict(torch.load("lstm.pth"))
model.eval()

targets, predictions = [], []
criterion = nn.MSELoss()
with torch.no_grad():
    predictions = model(test_data_features)
    targets = torch.flatten(test_data_targets)
    predictions = torch.flatten(predictions)
    rmse = torch.sqrt(criterion(torch.tensor(predictions[:len(predictions)-1]), torch.tensor(targets[:len(targets)-1])))
    print(f"RMSE: {rmse:.4f}")

predictions = denormalize(predictions, test_mean, test_std)
targets = denormalize(targets, test_mean, test_std)

# actual values are blue, predicted values are red
plt.plot(targets, c="b")
plt.plot(predictions, c="r")

targets_converted = convertToString(targets)
predictions_converted = convertToString(predictions)

print("targets")
print(targets_converted)

print("predictions")
print(predictions_converted)

print("\npredicted next banner --------------")
print(len(targets_converted[:len(targets_converted)-1]))
print(len(predictions_converted[:len(predictions_converted)-1]))
print(predictions_converted[-1])

plt.show()
