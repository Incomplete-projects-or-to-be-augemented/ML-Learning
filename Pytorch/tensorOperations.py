import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20, 30])

print(a + b)
print(a * b)

# Matrix muliplication:

A = torch.tensor([[1, 2],
                  [3, 4]])

B = torch.tensor([[5, 6],
                  [7, 8]])
print(A @ B)

a = torch.tensor([1, 2, 3])
b = 10

print(a+b)

x = torch.tensor([1.0, 4.0, 9.0])

sqrt = torch.sqrt(x)
mean = torch.mean(x)
