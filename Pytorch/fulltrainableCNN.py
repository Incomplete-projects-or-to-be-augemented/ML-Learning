import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- Model ----

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 32, 32)):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        flattened_size = self._get_flattened_size(input_shape)
        self.fc1 = nn.Linear(flattened_size, num_classes)

    def _get_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.pool1(self.relu1(self.conv1(dummy_input)))
            x = self.pool2(self.relu2(self.conv2(x)))
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# ---- Fake dataset (standing in for real images, just to see the mechanics work) ----

class FakeImageDataset(Dataset):
    def __init__(self, num_samples, num_classes, image_shape=(3, 32, 32)):
        self.images = torch.rand(num_samples, *image_shape)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

num_classes = 3
dataset = FakeImageDataset(num_samples=200, num_classes=num_classes)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---- Model, loss, optimizer ----

model = SimpleCNN(num_classes=num_classes)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---- Training loop (identical structure to the house price one) ----

num_epochs = 20

for epoch in range(num_epochs):
    for batch_images, batch_labels in data_loader:

        predictions = model(batch_images)

        loss = loss_function(predictions, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")