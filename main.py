from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd

# Read csv and load training data
class GachaDataset(Dataset):
    def __init__(self, datafile, labelfile, transform=None):
        self.data = pd.read_csv(datafile, usecols=["Double Banner", "Rerun Banner", "Element", "Weapon", "Archon", "Nation", "Build"])
        self.labels = pd.read_csv(labelfile)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)

        return data, label

genshin_data = GachaDataset("./data/genshin/genshin-data.csv", "./data/genshin/banner-names.csv")
test_genshin_data = GachaDataset("./data/genshin/test-genshin-data.csv", "./data/genshin/test-banner-names.csv")

train_dataloader = DataLoader(genshin_data, batch_size=64, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_genshin_data, batch_size=2, shuffle=False, num_workers=2)