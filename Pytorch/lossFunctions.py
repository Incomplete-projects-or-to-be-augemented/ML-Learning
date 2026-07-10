import torch
import torch.nn as nn

predictions = torch.tensor([3.0, 5.0, 2.5])
targets = torch.tensor([3.5, 5.0, 2.0])

loss_function = nn.MSELoss()

loss = loss_function(predictions, targets)

print(loss)