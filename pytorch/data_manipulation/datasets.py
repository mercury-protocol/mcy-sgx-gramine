import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np


class SplitMNISTDataSet(Dataset):
    def __init__(self, root, transform=None):
        self.images = self.read_idx3_file(f"{root}/train-images-idx3-ubyte")
        self.labels = self.read_idx1_file(f"{root}/train-labels-idx1-ubyte")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])  # Convert to integer
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def read_idx3_file(file_path):
        with open(file_path, 'rb') as f:
            # Skip the magic number and read metadata
            f.read(16)
            # Read image data
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the image data to a 3D array (num_images, num_rows, num_cols)
        num_images = len(image_data) // (28 * 28)
        image_data = image_data.reshape(num_images, 28, 28)
        return image_data

    @staticmethod
    def read_idx1_file(file_path):
        with open(file_path, 'rb') as f:
            # Skip the magic number and read metadata
            f.read(8)
            # Read label data
            label_data = np.frombuffer(f.read(), dtype=np.uint8)
        return label_data


if __name__ == "__main__":
    from pytorch.data_manipulation.constants import SPLIT_DATA_PATH, BATCH_SIZE_TRAIN, NORMALIZE_MEAN, NORMALIZE_STD
    from pytorch.helpers.eval import check_data

    custom_dataset = SplitMNISTDataSet(
        SPLIT_DATA_PATH + "part1",
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])
    )

    data_loader = torch.utils.data.DataLoader(
        custom_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True
    )

    check_data(data_loader)
