import torch
import torch.nn as nn
import torchvision

import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

from mcy_dist_ai.required_utils import DataSetFactory, DataLoaderFactory, NetworkFactory, OptimizerFactory


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
        return F.log_softmax(x)


# ------------------- create required objects ----------------
network_factory = NetworkFactory(ImageClassifier)
optimizer_factory = OptimizerFactory(SGD, lr=LEARNING_RATE, momentum=MOMENTUM)

data_set_factory = DataSetFactory(
    torchvision.datasets.MNIST,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

)

data_loader_factory = DataLoaderFactory(
    data_set_factory,
    DataLoader,
    batch_size=BATCH_SIZE,
    shuffle=True)


# ------------------- train the model ----------------
def train_batch(data, target, model, optimizer):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss  # for local logging purposes only


if __name__ == "__main__":
    data_loader = data_loader_factory.create("data")
    model = network_factory.create()
    optimizer = optimizer_factory.create(model.parameters())
    for epoch in range(N_EPOCHS):
        for batch_idx, (data, target) in enumerate(data_loader):
            train_batch(data, target, model, optimizer)

    from tests.tools import evaluate_model
    evaluate_model(model, "data")
