import torch
import torch.nn as nn

conv_layer = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)

sample_image = torch.rand(1, 3, 64, 64)  # (batch, channels, height, width)

output = conv_layer(sample_image)

print(output.shape)