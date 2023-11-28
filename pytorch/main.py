import os
import torch
from user_script import (
    train, data_loader_factory, network_factory, optimizer_factory,
    train_losses, train_counter
)
from pytorch.constants import SPLIT_DATA_PATH, NETWORK_PATH, OPTIMIZER_PATH
from pytorch.helpers.eval import evaluate_training, make_predictions
from pytorch.helpers.test_network import test, test_losses, test_counter, test_data_loader


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


def train_network():
    data_parts = os.listdir(SPLIT_DATA_PATH)
    data_loader = data_loader_factory.create(f"{SPLIT_DATA_PATH}/{data_parts[0]}/")

    network = load_network()
    test(data_loader, network, 0)

    for epoch in range(1, 3):
        network = load_network()
        optimizer = load_optimizer(network)

        train(data_loader, network, optimizer, epoch)
        test(data_loader, network, epoch)

        torch.save(network.state_dict(), NETWORK_PATH)
        torch.save(optimizer.state_dict(), OPTIMIZER_PATH)


def evaluate_network():
    network = load_network()
    evaluate_training(train_counter, train_losses, test_counter, test_losses)
    make_predictions(network, test_data_loader)


if __name__ == "__main__":
    train_network()
    evaluate_network()
