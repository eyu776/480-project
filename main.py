import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
from sklearn.metrics import accuracy_score

# read csv and load training data
class GachaDataset(Dataset):
    def __init__(self, datafile, labelfile):
        self.data = pd.read_csv(datafile, usecols=["double_banner", "rerun_banner", "element", "weapon", "archon", "nation", "build"])
        self.labels = pd.read_csv(labelfile, usecols=["num", "name"])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = [
            self.data.double_banner[idx],
            self.data.rerun_banner[idx],
            self.data.element[idx],
            self.data.weapon[idx],
            self.data.archon[idx],
            self.data.nation[idx],
            self.data.build[idx]
        ]
        label = self.labels.num[idx]
        
        # convert to tensor since we aren't using images
        data = torch.Tensor(data)
        data = nn.functional.normalize(data, dim=0)

        return data, label

genshin_data = GachaDataset("./data/genshin/genshin-data.csv", "./data/genshin/banner-names.csv")
test_genshin_data = GachaDataset("./data/genshin/test-genshin-data.csv", "./data/genshin/test-banner-names.csv")

num_workers = 0
train_dataloader = DataLoader(genshin_data, batch_size=64, shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(test_genshin_data, batch_size=2, shuffle=False, num_workers=num_workers)

# define neural network model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# define constants
input_size = 7
hidden_size = 49
output_size = 42
num_epochs = 500

# instantiate model
model = MLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
for epoch in range(num_epochs):
    model.train()
    for data, labels in train_dataloader:
        # forward
        outputs = model(data)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch[{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# testing
model.eval()
with torch.no_grad():
    correct = total = 0
    all_predictions, all_labels = [], []
    for data, labels in test_dataloader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted ==labels).sum().item()
        all_predictions += predicted.tolist()
        all_labels += labels.tolist()

    acurracy = 100 * correct / total
    print(f"Accuracy: {accuracy_score(all_labels, all_predictions) * 100:.2f}%")