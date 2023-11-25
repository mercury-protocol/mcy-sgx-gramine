import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


INITIAL_NETWORK_PATH = "../results/initial_network.pkl"
INITIAL_OPTIMIZER_PATH = "../results/initial_optimizer.pkl"
MODEL_PATH = "../results/model.pth"
OPTIMIZER_PATH = "../results/optimizer.pth"

LEARNING_RATE = 0.01
MOMENTUM = 0.5


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


# TODO: These are the requirements
initial_network = Net()
initial_optimizer = optim.SGD(initial_network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

with open(INITIAL_NETWORK_PATH, "wb") as file:
    pickle.dump(initial_network, file)
with open(INITIAL_OPTIMIZER_PATH, "wb") as file:
    pickle.dump(initial_optimizer, file)

torch.save(initial_network.state_dict(), MODEL_PATH)
torch.save(initial_optimizer.state_dict(), OPTIMIZER_PATH)
