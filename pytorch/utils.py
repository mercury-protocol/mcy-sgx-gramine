import os
import torch
from torch import nn

from user_script import network_factory, optimizer_factory


def load_network(path=""):
    network = network_factory.create()
    if os.path.exists(path):
        network.load_state_dict(torch.load(path))
    return network


def load_optimizer(network: nn.Module, path=""):
    optimizer = optimizer_factory.create(network.parameters())
    if os.path.exists(path):
        optimizer.load_state_dict(torch.load(path))
    return optimizer


def save_gradients(network: nn.Module, path):
    gradients = {name: param.grad.data for name, param in network.named_parameters()}
    torch.save(gradients, path)
