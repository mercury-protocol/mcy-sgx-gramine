import os
import torch
from torch import nn

from user_script import network_factory, optimizer_factory

from pytorch.constants import NETWORK_PATH, OPTIMIZER_PATH


def load_network(path=NETWORK_PATH):
    network = network_factory.create()
    if os.path.exists(path):
        network.load_state_dict(torch.load(path))
    return network


def load_optimizer(network: nn.Module, path=OPTIMIZER_PATH):
    optimizer = optimizer_factory.create(network.parameters())
    if os.path.exists(path):
        optimizer.load_state_dict(torch.load(path))
    return optimizer


def save_network_params(network: nn.Module, path):
    gradients = {name: p.grad.data for name, p in network.named_parameters()}
    with open(path, 'wb') as f:
        torch.save(gradients, f)