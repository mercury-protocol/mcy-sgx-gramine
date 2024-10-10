import importlib
import os
import torch
import struct

from collections.abc import Iterable
from pathlib import Path
from time import sleep
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from typing import Any, List, Union

from mcy_dist_ai.constants import (
    WAITING_PERIOD,
    USER_SCRIPT_PATH,
    CHECKPOINT_PATH,
    WORKER_NODES_NUM,
)
from mcy_dist_ai.logger import logger


script_path = os.path.abspath(USER_SCRIPT_PATH)
while not os.path.exists(script_path):
    sleep(WAITING_PERIOD)

sleep(2)  # TODO: use a safer method to wait if file is completely copied
spec = importlib.util.spec_from_file_location("user_script", script_path)
user_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(user_script)
logger.info("user_script.py imported.")


class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data = data_tensor
        self.targets = target_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def create_tensor_loader(data_path: str):
    data_tensor = torch.load(f"{data_path}/data_tensor.pt")
    target_tensor = torch.load(f"{data_path}/target_tensor.pt")

    tensor_dataset = TensorDataset(data_tensor, target_tensor)

    return DataLoader(
        tensor_dataset,
        batch_size=user_script.BATCH_SIZE,
        shuffle=True
    )


def torch_safe_load(path: Union[Path, str]) -> Any:
    for _ in range(10):
        try:
            return torch.load(path)
        except Exception as e:
            logger.warning(f"{type(e)}: {e}. Path: {path}. Size: {os.path.getsize(path) / 1024} KB.")
            sleep(2)

    raise Exception("Couldn't load file safely with torch.")


def load_model(path: Union[Path, str] = "", delete_file: bool = False) -> nn.Module:
    model = user_script.create_model()
    if os.path.exists(path):
        model.load_state_dict(torch_safe_load(path))
        if delete_file:
            os.remove(path)

    logger.debug("model loaded")
    return model


def load_optimizer(model: nn.Module) -> Any:
    optimizer = user_script.create_optimizer(model)
    return optimizer


def safe_create_extra_training_args(data_loader: DataLoader, optimizer: Optimizer) -> Iterable:
    extra_args = user_script.create_extra_training_args(data_loader, optimizer)
    if extra_args is None:
        return []
    if not isinstance(extra_args, Iterable):
        return [extra_args]
    return extra_args


def list_worker_nodes() -> List[str]:
    return [str(i + 1) for i in range(WORKER_NODES_NUM)]


def checkpoint(epoch: int, batch_idx: int):
    # we signal the next coming batch -> where work should be continued from
    checkpoint_data = struct.pack('!ii', epoch, batch_idx + 1)
    with open(CHECKPOINT_PATH, 'wb') as f:
        f.write(checkpoint_data)


def load_last_checkpoint() -> (int, int):
    if not os.path.exists(CHECKPOINT_PATH):
        logger.warning("checkpoint file does not exist - this is expected only before first worker iteration")
        return 0, 0

    with open(CHECKPOINT_PATH, 'rb') as f:
        checkpoint_data = f.read()
    epoch, batch = struct.unpack('!ii', checkpoint_data)

    return epoch, batch
