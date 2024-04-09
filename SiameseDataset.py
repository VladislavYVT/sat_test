from torch import tensor
from torchvision.io import read_image
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.data[idx]

        # Load images using OpenCV
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)

        # Apply transformations if specified
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        tens = tensor(float(label))
        return img1, img2, tens
