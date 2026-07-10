# 1. Forward pass    → model makes a prediction
# 2. Loss            → compare prediction to target
# 3. Zero gradients  → clear old gradients (remember accumulation!)
# 4. Backward pass   → compute new gradients
# 5. Optimizer step  → nudge weights using those gradients
# Press Cntrl + / to comment / uncomment multiple lines

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ---- Data ----

class HousePriceDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

raw_features = [
    [3.0, 2.0, 1500.0],
    [4.0, 3.0, 2200.0],
    [2.0, 1.0, 900.0],
    [5.0, 4.0, 3100.0],
]

raw_targets = [[410.0], [595.0], [250.0], [720.0]]

# ---- Scale features using scikit-learn ----

feature_scaler = StandardScaler()
scaled_features = feature_scaler.fit_transform(raw_features)

# Scale targets too -- this matters just as much as scaling features,
# since 410-720 is also a large, unscaled range compared to what
# small random weights naturally produce at the start
target_scaler = StandardScaler()
scaled_targets = target_scaler.fit_transform(raw_targets)


# ---- Convert to tensors ----

features = torch.tensor(scaled_features, dtype=torch.float32)
targets = torch.tensor(scaled_targets, dtype=torch.float32)

dataset = HousePriceDataset(features, targets)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# ---- Model ----

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

# ---- Loss + Optimizer ----

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100

for epoch in range(num_epochs):
    for batch_features, batch_targets in data_loader:

        predictions = model(batch_features)              # 1. forward pass

        loss = loss_function(predictions, batch_targets)  # 2. loss

        optimizer.zero_grad()                              # 3. zero gradients

        loss.backward()                                    # 4. backward pass

        optimizer.step()                                    # 5. optimizer step

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")