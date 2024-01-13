import os
import torch

from enum import Enum
from pathlib import Path
from torch import nn

from constants import WORKER_DIR
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


def list_worker_nodes() -> list[str]:
    return os.listdir(WORKER_DIR)


class FileEnum(Enum):
    STATE_DICT = "state_dict.pth"
    OPTIMIZER = "optimizer.pth"
    GRADIENT = "gradient.pth"
    BATCH_TRAINING_COMPLETE = "batch_training_complete"
    TRAINING_COMPLETE = "training_complete"
    BATCH_AGGREGATION_COMPLETE = "batch_aggregation_complete"


def get_file_path(node: str, filename: FileEnum) -> Path:
    return WORKER_DIR / str(node) / filename.value


def safe_delete_file(path):
    if os.path.exists(path):
        os.remove(path)
