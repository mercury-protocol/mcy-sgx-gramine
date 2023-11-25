from pytorch.normal.eval import evaluate_training, make_predictions

import torch
import torch.optim as optim
from user_script import (
    train,
    Network, LEARNING_RATE, MOMENTUM,
    test, test_loader, train_losses, train_counter, test_losses, test_counter
)
from pytorch.external_constants import NETWORK_PATH, OPTIMIZER_PATH


def load_network():
    # TODO: get network in a more general way
    network = Network()
    network.load_state_dict(torch.load(NETWORK_PATH))
    return network


def load_optimizer(network):
    # TODO: get optimizer in a more general way
    optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
    return optimizer


# ------------------- train the model ----------------
_network = load_network()
test(_network, 0)
for epoch in range(1, 3):
    _network = load_network()
    _optimizer = load_optimizer(_network)
    train(_network, _optimizer, epoch)
    test(_network, epoch)
    torch.save(_network.state_dict(), NETWORK_PATH)
    torch.save(_optimizer.state_dict(), OPTIMIZER_PATH)


# ------------------- evaluate the network ----------------
_network = load_network()
evaluate_training(train_counter, train_losses, test_counter, test_losses)
make_predictions(_network, test_loader)
