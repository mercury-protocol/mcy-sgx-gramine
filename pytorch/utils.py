import os
import torch

from enum import Enum
from pathlib import Path
from torch import nn

from constants import IO_DIR, SPLIT_DATA_PATH
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


class DirEnum(Enum):
    LEADER = "leader"
    WORKER = "worker"
    WATCHER = "watcher"


class FileEnum(Enum):
    STATE_DICT = "state_dict.pth"
    OPTIMIZER = "optimizer.pth"
    GRADIENT = "gradient.pth"
    BATCH_TRAINING_COMPLETE = "batch_training_complete"
    TRAINING_COMPLETE = "training_complete"
    BATCH_AGGREGATION_COMPLETE = "batch_aggregation_complete"
    AGGREGATION_COMPLETE = "aggregation_complete"


def get_file_path(directory: DirEnum, node: str, filename: FileEnum) -> Path:
    return IO_DIR / directory.value / str(node) / filename.value


def get_data_path(node: str) -> Path:
    return SPLIT_DATA_PATH / str(node)


def safe_delete_file(path):
    if os.path.exists(path):
        os.remove(path)
