import torch
import torchvision

from pytorch.helpers.eval import check_data
from pytorch.data_manipulation.constants import (
    BATCH_SIZE_TRAIN, BATCH_SIZE_TEST, RANDOM_SEED, NORMALIZE_MEAN, NORMALIZE_STD
)


torch.backends.cudnn.enabled = False
torch.manual_seed(RANDOM_SEED)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("mnist_digits", train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
                               ])),
    batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("mnist_digits", train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
                               ])),
    batch_size=BATCH_SIZE_TEST, shuffle=True)


if __name__ == "__main__":
    check_data(test_loader)
