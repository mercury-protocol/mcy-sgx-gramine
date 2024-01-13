import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from pytorch.data_manipulation.datasets import SplitMNISTDataSet  # this will be implemented by the user
from pytorch.required_utils import DataSetFactory, DataLoaderFactory, NetworkFactory, OptimizerFactory


# ------------------- config ----------------
N_EPOCHS = 2
BATCH_SIZE_TRAIN = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.5

RANDOM_SEED = 1

torch.backends.cudnn.enabled = False
torch.manual_seed(RANDOM_SEED)


# ------------------- build the network ----------------
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
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
network_factory = NetworkFactory(Network)
optimizer_factory = OptimizerFactory(optim.SGD, lr=LEARNING_RATE, momentum=MOMENTUM)

data_set_factory = DataSetFactory(
    SplitMNISTDataSet,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
)
data_loader_factory = DataLoaderFactory(
    data_set_factory,
    torch.utils.data.DataLoader,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True)


# ------------------- train the model ----------------
def train_batch(data, target, network, optimizer):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss  # for local logging purposes only
