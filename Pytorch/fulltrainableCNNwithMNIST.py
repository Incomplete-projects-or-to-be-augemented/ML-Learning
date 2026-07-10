import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---- Load real data ----

transform = transforms.ToTensor()  # converts images to tensors, scales pixel values to [0, 1]

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ---- Model ----

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_shape=(1, 28, 28)):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
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

model = SimpleCNN(num_classes=10)  # 10 digits: 0-9

# ---- Loss + optimizer ----

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---- Training loop ----

num_epochs = 3

for epoch in range(num_epochs):
    total_loss = 0
    for batch_images, batch_labels in train_loader:

        predictions = model(batch_images)

        loss = loss_function(predictions, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Average Loss: {average_loss:.4f}")