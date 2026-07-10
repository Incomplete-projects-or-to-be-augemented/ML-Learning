import torch
import torch.nn as nn

layer = nn.Linear(in_features=3, out_features=1)

sample_input = torch.tensor([1.0, 2.0, 3.0])

output = layer(sample_input)

print(output)

print(layer.weight)
print(layer.bias)


## Multiple layers


import torch
import torch.nn as nn

layer1 = nn.Linear(3, 4)
activation = nn.ReLU()
layer2 = nn.Linear(4, 1)

sample_input = torch.tensor([1.0, 2.0, 3.0])

hidden = layer1(sample_input)
activated = activation(hidden)
output = layer2(activated)

print(output)