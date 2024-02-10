import importlib
import os
import torch

from time import sleep
from torch import nn
from typing import List

from pytorch.constants import WAITING_PERIOD, WORKER_NODES_NUM, USER_SCRIPT_PATH
from pytorch.logger import logger


script_path = os.path.abspath(USER_SCRIPT_PATH)
while not os.path.exists(script_path):
    sleep(WAITING_PERIOD)

spec = importlib.util.spec_from_file_location("user_script", script_path)
user_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(user_script)
logger.info("user_script.py imported.")


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
    return [str(i) for i in range(WORKER_NODES_NUM)]
