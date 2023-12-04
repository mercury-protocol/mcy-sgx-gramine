import os
import torch

from user_script import train, data_loader_factory, network_factory, optimizer_factory

from pytorch.constants import (
    SPLIT_DATA_PATH, NETWORK_PATH, OPTIMIZER_PATH, SPLIT_NETWORK_PATH, SPLIT_OPTIMIZER_PATH, AGGREGATED_NETWORK_PATH
)
from pytorch.helpers.eval import make_predictions
from pytorch.helpers.test_network import test, test_data_loader


def load_network(path=NETWORK_PATH):
    network = network_factory.create()
    if os.path.exists(path):
        network.load_state_dict(torch.load(path))
    return network


def load_optimizer(network, path=OPTIMIZER_PATH):
    optimizer = optimizer_factory.create(network.parameters())
    if os.path.exists(path):
        optimizer.load_state_dict(torch.load(path))
    return optimizer


def train_network():
    for dirname in os.listdir(SPLIT_DATA_PATH):
        data_path = f"{SPLIT_DATA_PATH}/{dirname}/"
        network_path = f"{SPLIT_NETWORK_PATH}/{dirname}/network.pth"
        optimizer_path = f"{SPLIT_OPTIMIZER_PATH}/{dirname}/optimizer.pth"

        data_loader = data_loader_factory.create(data_path)
        network = load_network(path=network_path)
        optimizer = load_optimizer(network, path=optimizer_path)

        train(data_loader, network, optimizer)

        torch.save(network.state_dict(), network_path)
        torch.save(optimizer.state_dict(), optimizer_path)


def evaluate_network():
    network = load_network(AGGREGATED_NETWORK_PATH)
    test(network)
    make_predictions(network, test_data_loader)


if __name__ == "__main__":
    train_network()
    evaluate_network()
