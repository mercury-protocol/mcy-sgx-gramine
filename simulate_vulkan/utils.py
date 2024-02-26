import os
import torch
from pathlib import Path
from typing import List

from pytorch.constants import LEADER_DIR

from simulate_vulkan import user_script
from simulate_vulkan.constants import WORKER_NODES_NUM


def load_network(path=""):
    network = user_script.network_factory.create()
    if os.path.exists(path):
        network.load_state_dict(torch.load(path))
    return network


def list_worker_nodes() -> List[str]:
    return [str(i + 1) for i in range(WORKER_NODES_NUM)]


def leader_get_path(worker_node: str, file: str) -> Path:
    if "." in file:
        name, extension = file.split(".")
        file = f"{name}_{worker_node}.{extension}"
    else:
        file = f"{file}_{worker_node}"
    return LEADER_DIR / file
