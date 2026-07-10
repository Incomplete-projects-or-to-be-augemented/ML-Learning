import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class HousePriceDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]
    
features = torch.tensor([
    [3.0, 2.0, 1500.0],
    [4.0, 3.0, 2200.0],
    [2.0, 1.0, 900.0],
    [5.0, 4.0, 3100.0],
])

targets = torch.tensor([410.0, 595.0, 250.0, 720.0])

dataset = HousePriceDataset(features, targets)

print(len(dataset))
print(dataset[0])


data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_features, batch_targets in data_loader:
    print(batch_features)
    print(batch_targets)
    print("---")