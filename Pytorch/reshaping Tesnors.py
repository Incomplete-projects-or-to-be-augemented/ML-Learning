import torch


x = torch.tensor([1,2,3,4,5,6])

print(x)

# Element overall shuold remain same
y = x.reshape(2,3)

print(y)

z = torch.arange(12)


print(z)
