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


l = torch.rand(32, 3, 224, 224)

flat = rearrange(
    l,
    'b c h w -> b (c h w)'
)

print(flat)

# ============================================
# RESHAPE
# ============================================

print("----- RESHAPE -----")

# Create a 1D tensor with 12 numbers
original_tensor = torch.arange(12)

print("Original tensor:")
print(original_tensor)

print("Original shape:")
print(original_tensor.shape)


# Change the grouping of the numbers
# 12 elements -> 3 rows of 4 numbers
reshaped_tensor = original_tensor.reshape(3, 4)

print("\nReshaped tensor:")
print(reshaped_tensor)

print("Reshaped shape:")
print(reshaped_tensor.shape)



# ============================================
# VIEW
# ============================================

print("\n----- VIEW -----")

# view() does the same shape change if the memory layout allows it

viewed_tensor = original_tensor.view(3, 4)

print("Viewed tensor:")
print(viewed_tensor)

print("Viewed shape:")
print(viewed_tensor.shape)



# ============================================
# PERMUTE
# ============================================

print("\n----- PERMUTE -----")

# Create an image tensor
# Shape:
# channels, height, width

image_tensor = torch.rand(3, 224, 224)

print("Original image shape:")
print(image_tensor.shape)


# Move channels from the first dimension to the last dimension
# Before:
# (channels, height, width)
#
# After:
# (height, width, channels)

permuted_image_tensor = image_tensor.permute(1, 2, 0)

print("\nPermuted image shape:")
print(permuted_image_tensor.shape)



# ============================================
# REARRANGE
# ============================================

print("\n----- REARRANGE -----")

# Rearrange using named dimensions

rearranged_image_tensor = rearrange(
    image_tensor,
    'channels height width -> height width channels'
)

print("Rearranged image shape:")
print(rearranged_image_tensor.shape)



# ============================================
# RESHAPE WITH IMAGES
# ============================================

print("\n----- FLATTEN IMAGE -----")

# Batch of images
# Shape:
# batch, channels, height, width

batch_images_tensor = torch.rand(32, 3, 224, 224)

print("Original batch shape:")
print(batch_images_tensor.shape)


# Flatten every image while keeping the batch dimension

flattened_images_tensor = rearrange(
    batch_images_tensor,
    'batch channels height width -> batch (channels height width)'
)

print("\nFlattened batch shape:")
print(flattened_images_tensor.shape)