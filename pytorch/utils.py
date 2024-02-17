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

sleep(2)  # TODO: use a safer method to wait if file is completely copied
spec = importlib.util.spec_from_file_location("user_script", script_path)
user_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(user_script)
logger.info("user_script.py imported.")
logger.info(f'user_script has network_factory: {hasattr(user_script, "network_factory")}')


def torch_safe_load(path):
    for _ in range(10):
        try:
            return torch.load(path)
        except Exception as e:
            logger.warning(f"{type(e)}: {e}. Path: {path}. Size: {os.path.getsize(path) / 1024} KB.")
            sleep(2)

    raise Exception("Couldn't load file safely with torch.")


def load_network(path="", delete_file=False):
    network = user_script.network_factory.create()
    if os.path.exists(path):
        network.load_state_dict(torch_safe_load(path))
        if delete_file:
            os.remove(path)

    logger.debug("network loaded")
    return network


def load_optimizer(network: nn.Module):
    optimizer = user_script.optimizer_factory.create(network.parameters())
    return optimizer


def list_worker_nodes() -> List[str]:
    return [str(i) for i in range(WORKER_NODES_NUM)]
