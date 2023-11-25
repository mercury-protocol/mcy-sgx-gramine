from pytorch.normal.eval import evaluate_training, make_predictions
from pytorch.external_constants import DATA_PATH, MODEL_PATH, OPTIMIZER_PATH


# ------------------- get the data ----------------
import torch
import torchvision


N_EPOCHS = 2
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10

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


# ------------------- build the network ----------------
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


def load_network():
    network = Net()
    network.load_state_dict(torch.load(MODEL_PATH))
    return network


def load_optimizer(network):
    optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
    return optimizer


initial_network = Net()
initial_optimizer = optim.SGD(initial_network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

torch.save(initial_network.state_dict(), MODEL_PATH)
torch.save(initial_optimizer.state_dict(), OPTIMIZER_PATH)


# ------------------- train the model ----------------
train_losses = []
train_counter = []
test_losses = []
test_counter = []


def train(epoch):
    network = load_network()
    optimizer = load_optimizer(network)

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), MODEL_PATH)
            torch.save(optimizer.state_dict(), OPTIMIZER_PATH)

    torch.save(network.state_dict(), MODEL_PATH)
    torch.save(optimizer.state_dict(), OPTIMIZER_PATH)


def test(epoch):
    network = load_network()

    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_counter.append(epoch * len(train_loader.dataset))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test(0)
for e in range(1, N_EPOCHS + 1):
    train(e)
    test(e)


# ------------------- evaluate the model ----------------
network_under_training = load_network()
evaluate_training(train_counter, train_losses, test_counter, test_losses)
make_predictions(network_under_training, test_loader)


# ------------------- continued training from checkpoints ----------------
for e in range(N_EPOCHS + 1, N_EPOCHS + 2):
    train(e)
    test(e)


# ------------------- evaluate the better model ----------------
final_network = load_network()
evaluate_training(train_counter, train_losses, test_counter, test_losses)
make_predictions(final_network, test_loader)
