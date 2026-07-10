import torch
import torch.nn as nn

pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

sample_input = torch.rand(1, 8, 64, 64)  # (batch, channels, height, width)

output = pool_layer(sample_input)

print(output.shape)

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        print("After block 1:", x.shape)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        print("After block 2:", x.shape)

        x = self.flatten(x)
        print("After flatten:", x.shape)

        x = self.fc1(x)
        print("After fc1 (final output):", x.shape)

        return x

model = SimpleCNN(num_classes=3)

sample_image = torch.rand(1, 3, 64, 64)

output = model(sample_image)