import torch


x = torch.tensor([1, 2, 3, 4])

print("\nRegular 1D tensor: \n ")
print(x)
print("\nRegular 1D tensor's shape: \n ")
print(x.shape)

y = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print("\nRegular 2D tensor: \n ")
print(y)
print("\nRegular 2D tensor's shape: \n ")
print(y.shape)

z = torch.tensor([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])

print("\nRegular 3D tensor: \n ")
print(z)
print("\nRegular 3D tensor's shape: \n ")
print(z.shape)

a = torch.zeros(2, 3, dtype=torch.int)

print("\n2D tensor of zeroes: \n ")
print(a)
print("\n2D tensor of zeroes' shape: \n ")
print(a.shape)

b = torch.ones(2, 3, dtype=torch.int)

print("\n2D tensor of ones: \n ")
print(b)
print("\n2D tensor of ones' shape: \n ")
print(b.shape)

c = torch.rand(2, 3)

print("\nRandom 2D tensor: \n ")
print(c)
print("\nRandom 2D tensor's shape: \n ")
print(c.shape)