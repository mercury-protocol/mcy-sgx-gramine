import torch
import torch.nn as nn
import torchvision

import torch.nn.functional as F
from torch.optim import SGD
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
def create_model():
    return ImageClassifier()


def create_optimizer(model: nn.Module):
    return SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


def create_data_loader(path):
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


# ------------------- train the model ----------------
def train_batch(data, target, model, optimizer):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss  # for local logging purposes only


if __name__ == "__main__":
    # Test set: Avg. loss: 0.1994, Accuracy: 9408/10000 (94%)

    data_loader = create_data_loader("data")
    model = create_model()
    optimizer = create_optimizer(model)
    for epoch in range(N_EPOCHS):
        for batch_idx, (data, target) in enumerate(data_loader):
            train_batch(data, target, model, optimizer)

    from tests.utils import evaluate_model
    evaluate_model(model, "data")
