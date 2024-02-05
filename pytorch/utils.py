import os
import torch

from torch import nn

from constants import SPLIT_DATA_PATH
from user_script import network_factory, optimizer_factory


def load_network(path="", delete_file=False):
    network = network_factory.create()
    if os.path.exists(path):
        network.load_state_dict(torch.load(path))
        if delete_file:
            os.remove(path)
    return network


def load_optimizer(network: nn.Module):
    optimizer = optimizer_factory.create(network.parameters())
    return optimizer


def list_worker_nodes() -> list[str]:
    return os.listdir(SPLIT_DATA_PATH)
