import torch

x = torch.tensor([10, 20, 30, 40])

print(x.shape)

print(x[0])

print(x[1])

print(x[1:3])

print((x[1:3]).shape)

matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print(matrix.shape)

print(matrix[0])

print((matrix[0]).shape)


print(matrix[0, 1])

print(matrix[1, 2])