import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np

# read csv
train_data_csv = pd.read_csv("./data/genshin/genshin-data.csv", usecols=["double_banner", "rerun_banner", "element", "weapon", "archon", "nation", "build"])
# train_labels_csv = pd.read_csv("./data/genshin/banner-names.csv", usecols=["num", "name"])
test_data_csv = pd.read_csv("./data/genshin/test-genshin-data.csv", usecols=["double_banner", "rerun_banner", "element", "weapon", "archon", "nation", "build"])
# test_labels_csv = pd.read_csv("./data/genshin/test-banner-names.csv", usecols=["num", "name"])

# convert data to array
train_data_csv = train_data_csv.to_numpy().astype("float32")
test_data_csv = test_data_csv.to_numpy().astype("float32")

# create tensors of the data and the window to lookback in for our LSTM
def create_dataset(data, window):
    features, targets = [], []
    for i in range(len(data) - window):
        feat = data[i:i+window]
        target = data[i+1:i+window+1]
        features.append(feat)
        targets.append(target)

    features_tensor = torch.tensor(features)
    features_tensor = nn.functional.normalize(features_tensor, dim=0)
    targets_tensor = torch.tensor(targets)
    targets_tensor = nn.functional.normalize(targets_tensor, dim=0)

    return features_tensor, targets_tensor

# we don't shuffle with time series prediction problems
window = 4
train_data_features, train_data_targets = create_dataset(train_data_csv, window=window)
test_data_features, test_data_targets = create_dataset(test_data_csv, window=window)

# print(train_data_features.shape, train_data_targets.shape)
# print(test_data_features.shape, test_data_targets.shape)

# define neural network model
# using a long short term memory model since to predict the immediate next banner, i need all of the previous data
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc1 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # we don't need to know the LSTM's memory so we can discard the second output
        x, _ = self.lstm(x)
        x = self.fc1(x)
        return x
    
# define constants
input_size = 7
hidden_size = 49
output_size = 7
num_epochs = 200
num_layers = 7
batch_first = True

# instantiate model
model = LSTM_Model(input_size, hidden_size, num_layers, batch_first, output_size)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# load data
train_dataloader = DataLoader(TensorDataset(train_data_features, train_data_targets), shuffle=True, batch_size=8)
test_dataloader = DataLoader(TensorDataset(test_data_features, test_data_targets), shuffle=False, batch_size=8)

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

# testing
model.eval()
with torch.no_grad():
    correct = total = 0
    all_predictions, all_targets = [], []
    for features, targets in test_dataloader:
        predicted = model(features)
        # _, predicted = torch.max(output.data, 1)
        # total += targets.size(0)

        predicted = torch.flatten(predicted)
        targets = torch.flatten(targets)
        # print(f"predicted: {predicted}")

        # correct += (predicted == targets).sum().item()
        all_predictions += predicted.tolist()
        all_targets += targets.tolist()

    # accuracy = 100 * correct / total
    # print(f"all_predictions: {all_predictions}")
    # print(f"all_targets: {all_targets}")
    print(f"all_predictions.shape: {torch.tensor(all_predictions).shape}")
    print(f"all_targets.shape: {torch.tensor(all_targets).shape}")
    print("all_predictions")
    print(torch.tensor(all_predictions))
    print("all_targets")
    print(torch.tensor(all_targets))
    # print(f"Accuracy: {accuracy_score(all_targets, all_predictions) * 100:.2f}%")
    rmse = torch.sqrt(criterion(torch.tensor(all_predictions), torch.tensor(all_targets)))
    print(f"RMSE: {rmse:.2f}")