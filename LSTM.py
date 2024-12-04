import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read csv
data_csv = pd.read_csv("./data/enstars/enstars_data.csv", usecols=["banner"])

# convert data to array
data_csv = data_csv.to_numpy().astype("float32")

# create tensors of the data and the window to lookback in for our LSTM
def create_dataset(data, window):
    features, targets = [], []
    for i in range(len(data) - window):
        feat = data[i:i+window]
        target = data[i+1:i+window+1]
        features.append(feat)
        targets.append(target)

    features_tensor = torch.tensor(features)
    targets_tensor = torch.tensor(targets)

    return features_tensor, targets_tensor

# manual normalization
def normalize(arr, mean, std):
    for i, n in enumerate(arr):
        arr[i] = (n - mean) / std

# manual denormalization 
def denormalize(arr, mean, std):
    for i, n in enumerate(arr):
        arr[i] = (n * std) + mean

# we don't shuffle with time series prediction problems
window = 2
train_size = int(len(data_csv) * 0.8)
train_data_csv = data_csv[:train_size]
test_data_csv = data_csv[train_size:]

train_mean, train_std = train_data_csv.mean().item(), train_data_csv.std().item()
test_mean, test_std = test_data_csv.mean().item(), test_data_csv.std().item()

normalize(train_data_csv, train_mean, train_std)
normalize(test_data_csv, test_mean, test_std)

train_data_features, train_data_targets = create_dataset(train_data_csv, window=window)
test_data_features, test_data_targets = create_dataset(test_data_csv, window=window)

# define neural network model
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc1 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x)
        return x
    
# define constants
input_size = 1
hidden_size = 37
output_size = 1
num_epochs = 350
num_layers = 2
batch_first = True

# instantiate model
model = LSTM_Model(input_size, hidden_size, num_layers, batch_first, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# load data
train_dataloader = DataLoader(TensorDataset(train_data_features, train_data_targets), shuffle=True, batch_size=64)
test_dataloader = DataLoader(TensorDataset(test_data_features, test_data_targets), shuffle=False, batch_size=64)

# training
for epoch in range(num_epochs):
    model.train()
    for features, targets in train_dataloader:
        # forward
        output = model(features)
        loss = criterion(output, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

all_predictions, all_targets = [], []
model.eval()
with torch.no_grad():
    all_predictions = model(test_data_features)
    all_predictions = torch.flatten(all_predictions)
    all_targets = torch.flatten(test_data_targets)
    rmse = torch.sqrt(criterion(torch.tensor(all_predictions), torch.tensor(all_targets)))

torch.save(model.state_dict(), "lstm.pth")