# for all imports
from PIL import Image
from torch.utils.data import Dataset
# labelled images imports 
from torchvision import transforms


# unlabbled images imports
import os

# csv file imports
import os
import pandas as pd


# labelled images preprocessing into tensors
image = Image.open(r"C:\Users\khait\Downloads\images.jpeg")   # opens as a PIL Image, not a tensor yet

transform = transforms.Compose([
    transforms.Resize((64, 64)),      # force a consistent size
    transforms.ToTensor(),             # convert to tensor, scale pixels to [0,1]
])

image_tensor = transform(image)

print(image_tensor.shape)

# for a folder of images

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=r"C:\Users\khait\Downloads\images", transform=transform)

data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_images, batch_labels in data_loader:
    print(batch_images.shape)
    print(batch_labels)
    break



# unlabelled images preprocessing into tensors

class UnlabeledImageDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.folder_path = folder_path
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")

        self.image_filenames = [
            filename for filename in os.listdir(folder_path)
            if filename.lower().endswith(valid_extensions)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = os.path.join(self.folder_path, self.image_filenames[index])
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)
    
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = UnlabeledImageDataset(
    folder_path=r"C:\Users\khait\Downloads\images",
    transform=transform
)

data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_images in data_loader:
    print(batch_images.shape)   # e.g. torch.Size([16, 3, 64, 64])
    break


# csv file preprocessing images
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CSVLabeledImageDataset(Dataset):
    def __init__(self, folder_path, csv_path, transform):
        self.folder_path = folder_path
        self.transform = transform

        self.labels_df = pd.read_csv(csv_path)

        # turn text labels ("cat", "dog") into integer indices (0, 1)
        self.classes = sorted(self.labels_df["label"].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        row = self.labels_df.iloc[index]

        image_path = os.path.join(self.folder_path, row["filename"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        label = self.class_to_idx[row["label"]]

        return image_tensor, label


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = CSVLabeledImageDataset(
    folder_path=r"C:\Users\khait\Downloads\images",
    csv_path=r"C:\Users\khait\Downloads\images\labels.csv",
    transform=transform
)

data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

print(dataset.classes)          # ['cat', 'dog']
print(dataset.class_to_idx)     # {'cat': 0, 'dog': 1}

for batch_images, batch_labels in data_loader:
    print(batch_images.shape)   # torch.Size([16, 3, 64, 64])
    print(batch_labels)         # e.g. tensor([0, 1, 0, 0, 1, ...])
    break