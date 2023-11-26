import os
import torch
from user_script import (
    train, network_factory, optimizer_factory,
    test, test_loader, train_losses, train_counter, test_losses, test_counter
)
from pytorch.external_constants import NETWORK_PATH, OPTIMIZER_PATH
from pytorch.normal.eval import evaluate_training, make_predictions


def load_network():
    network = network_factory.create()
    if os.path.exists(NETWORK_PATH):
        network.load_state_dict(torch.load(NETWORK_PATH))
    return network


def load_optimizer(network):
    optimizer = optimizer_factory.create(network.parameters())
    if os.path.exists(OPTIMIZER_PATH):
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
