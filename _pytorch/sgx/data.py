import torch
import torchvision


DATA_PATH = "../mnist_digits/"

BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000

RANDOM_SEED = 1

torch.backends.cudnn.enabled = False
torch.manual_seed(RANDOM_SEED)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(DATA_PATH, train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(DATA_PATH, train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=BATCH_SIZE_TEST, shuffle=True)
