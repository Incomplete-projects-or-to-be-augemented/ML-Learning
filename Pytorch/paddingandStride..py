import torch
import torch.nn as nn

sample_image = torch.rand(1, 3, 64, 64)  # (batch, channels, height, width)

# ---- No padding, stride 1 (what we did originally) ----
conv_basic = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
output_basic = conv_basic(sample_image)
print("No padding, stride 1:", output_basic.shape)

# ---- Padding to preserve size ----
conv_same_size = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
output_same_size = conv_same_size(sample_image)
print("Padding=1, stride 1:", output_same_size.shape)

# ---- Stride 2, with padding (the case we just calculated by hand) ----
conv_strided = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, stride=2)
output_strided = conv_strided(sample_image)
print("Padding=1, stride 2:", output_strided.shape)