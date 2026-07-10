import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        print("After conv1:", x.shape)

        x = self.conv2(x)
        x = self.relu2(x)
        print("After conv2:", x.shape)

        x = self.conv3(x)
        x = self.relu3(x)
        print("After conv3:", x.shape)

        return x

model = SimpleCNN()

sample_image = torch.rand(1, 3, 64, 64)

output = model(sample_image)