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
data_csv = pd.read_csv("./data/enstars/enstars_data_train.csv", usecols=["banner"])

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
    features_tensor = nn.functional.normalize(features_tensor, dim=0)
    targets_tensor = torch.tensor(targets)
    targets_tensor = nn.functional.normalize(targets_tensor, dim=0)

    return features_tensor, targets_tensor

# # manual normalization
# def normalize(arr, b, a):
#     for i, n in enumerate(arr):
#         arr[i] = (n - a) / (b - a)

# # manual denormalization 
# def denormalize(arr, b, a):
#     for i, n in enumerate(arr):
#         arr[i] = a + (b - a) * n

# # manual normalization
# def normalize(arr, mean, std):
#     for i, n in enumerate(arr):
#         arr[i] = (n - mean) / std

# # manual denormalization 
# def denormalize(arr, mean, std):
#     for i, n in enumerate(arr):
#         arr[i] = (n * std) + mean

# we don't shuffle with time series prediction problems
window = 3
train_size = int(len(data_csv) * 0.67)
# train_data_csv = data_csv[:train_size]
# test_data_csv = data_csv[train_size:]

# train_max, train_min = train_data_csv.max(), train_data_csv.min()
# test_max, test_min = test_data_csv.max(), test_data_csv.min()
# max, min = data_csv.max(), data_csv.min()
# mean, std = data_csv.mean().item(), data_csv.std().item()
# train_mean, train_std = train_data_csv.mean().item(), train_data_csv.std().item()
# test_mean, test_std = test_data_csv.mean().item(), test_data_csv.std().item()

# normalize(train_data_csv, train_max, train_min)
# normalize(test_data_csv, test_max, test_min)
# normalize(data_csv, max, min)
# normalize(data_csv, mean, std)
# normalize(train_data_csv, mean, std)
# normalize(test_data_csv, mean, std)

train_data_csv = data_csv[:train_size]
test_data_csv = data_csv[train_size:]
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
input_size = 1
# hidden_size = 37
hidden_size = 256
output_size = 1
# num_epochs = 350
num_epochs = 500
# num_layers = 2
num_layers = 4
batch_first = True

# instantiate model
model = LSTM_Model(input_size, hidden_size, num_layers, batch_first, output_size)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001)

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

        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(train_data_features)
            train_rmse = np.sqrt(criterion(y_pred, train_data_targets))
            y_pred = model(test_data_features)
            test_rmse = np.sqrt(criterion(y_pred, test_data_targets))
            print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

all_predictions, all_targets = [], []
model.eval()
with torch.no_grad():
    all_predictions = model(test_data_features)
    all_predictions = torch.flatten(all_predictions)
    all_targets = torch.flatten(test_data_targets)
    rmse = torch.sqrt(criterion(torch.tensor(all_predictions), torch.tensor(all_targets)))

    # for features, targets in test_dataloader:
    #     predicted = model(features)

    #     predicted = torch.flatten(predicted)
    #     targets = torch.flatten(targets)

    #     all_predictions += predicted.tolist()
    #     all_targets += targets.tolist()

    # print("all_targets")
    # print(torch.tensor(all_targets))
    # print("all_predictions")
    # print(torch.tensor(all_predictions))
    # rmse = torch.sqrt(criterion(torch.tensor(all_predictions), torch.tensor(all_targets)))
    print(f"RMSE: {rmse:.4f}")

# denormalize(all_targets, mean, std)
# denormalize(all_predictions, mean, std)

plt.plot(all_targets, c="b")
plt.plot(all_predictions, c="r")

# print("all_targets")
# print(torch.tensor(all_targets))
# print("\nall_predictions")
# print(torch.tensor(all_predictions))

plt.show()

# torch.save(model.state_dict(), "lstm.pth")