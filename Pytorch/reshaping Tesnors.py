import torch
from einops import rearrange

x = torch.tensor([1,2,3,4,5,6])

print(x)

# Element overall shuold remain same
y = x.reshape(2,3)

print(y)

z = torch.arange(12)


print(z)


p = z.reshape(3,4)

print(p)

image = torch.rand(3, 224, 224)

image_new = rearrange(
    image,
    'c h w -> h w c'
)

print(image_new)