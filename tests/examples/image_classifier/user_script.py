import torch
import torch.nn as nn
import torchvision

import torch.nn.functional as F
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader


# ------------------- config ----------------
N_EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.5

RANDOM_SEED = 1

torch.backends.cudnn.enabled = False
torch.manual_seed(RANDOM_SEED)


# ------------------- build the network ----------------
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ------------------- create required objects ----------------
def create_model() -> nn.Module:
    return ImageClassifier()


def create_optimizer(model: nn.Module) -> Optimizer:
    return SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


def create_data_loader(path) -> DataLoader:
    # if download returns 403, clode this:
    # https://github.com/knamdar/data/tree/master
    dataset = torchvision.datasets.MNIST(
        path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


def create_extra_training_args(data_loader: DataLoader, optimizer: Optimizer):
    pass


# ------------------- train the model ----------------
def train_batch(batch, model, optimizer):
    data, target = batch
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss  # for local logging purposes only


# ---------- for testing purpose ----------
def create_eval_data_loader(path) -> DataLoader:
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(path, train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=1000, shuffle=True)
