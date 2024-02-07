import importlib
import os
import torch

from time import sleep
from torch import nn
from typing import List

from constants import SPLIT_DATA_PATH, WAITING_PERIOD

while not os.path.exists("user_script.py"):  # TODO: import from /io
    sleep(WAITING_PERIOD)
user_script = importlib.import_module("user_script")


def load_network(path="", delete_file=False):
    network = user_script.network_factory.create()
    if os.path.exists(path):
        network.load_state_dict(torch.load(path))
        if delete_file:
            os.remove(path)
    return network


def load_optimizer(network: nn.Module):
    optimizer = user_script.optimizer_factory.create(network.parameters())
    return optimizer


def list_worker_nodes() -> List[str]:
    return os.listdir(SPLIT_DATA_PATH)  # TODO: pass worker nodes to leader container
