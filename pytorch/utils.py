import argparse
import importlib
import os
import torch
import sys
import struct

from pathlib import Path
from time import sleep
from torch import nn
from typing import Any, List, Union

import pytorch.constants as constants

from pytorch.constants import (
    WAITING_PERIOD,
    LEADER_ROLE,
    WORKER_ROLE,
    WORKER_LLM_ROLE,
    USER_SCRIPT_PATH,
    CHECKPOINT_PATH,
    set_role_and_worker_node_num
)
from pytorch.logger import logger


parser = argparse.ArgumentParser()
parser.add_argument("--role", type=str, help="Node role - leader, worker or worker-llm")
parser.add_argument("--worker_count", type=int, help="Worker nodes count")
args = parser.parse_args()
if args.role is None:
    logger.error("Role argument is missing")
    sys.exit(1)
elif args.role.upper() not in (LEADER_ROLE, WORKER_ROLE, WORKER_LLM_ROLE):
    logger.error(f"role must be {LEADER_ROLE}, {WORKER_ROLE} or {WORKER_LLM_ROLE}")
    sys.exit(1)

if args.role == LEADER_ROLE and args.worker_count is None:
    logger.error("Worker nodes count argument is required for leader")
    sys.exit(1)

set_role_and_worker_node_num(role=args.role, worker_nodes_num=args.worker_count)

script_path = os.path.abspath(USER_SCRIPT_PATH)
while not os.path.exists(script_path):
    sleep(WAITING_PERIOD)

sleep(2)  # TODO: use a safer method to wait if file is completely copied
spec = importlib.util.spec_from_file_location("user_script", script_path)
user_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(user_script)
logger.info("user_script.py imported.")
logger.info(f'user_script has network_factory: {hasattr(user_script, "network_factory")}')


def torch_safe_load(path: Union[Path, str]) -> Any:
    for _ in range(10):
        try:
            return torch.load(path)
        except Exception as e:
            logger.warning(f"{type(e)}: {e}. Path: {path}. Size: {os.path.getsize(path) / 1024} KB.")
            sleep(2)

    raise Exception("Couldn't load file safely with torch.")


def load_network(path: Union[Path, str] = "", delete_file: bool = False) -> nn.Module:
    network = user_script.network_factory.create()
    if os.path.exists(path):
        network.load_state_dict(torch_safe_load(path))
        if delete_file:
            os.remove(path)

    logger.debug("network loaded")
    return network


def load_optimizer(network: nn.Module) -> Any:
    optimizer = user_script.optimizer_factory.create(network.parameters())
    return optimizer


def list_worker_nodes() -> List[str]:
    return [str(i + 1) for i in range(constants.WORKER_NODES_NUM)]


def checkpoint(epoch: int, batch_idx: int):
    checkpointed_idx = batch_idx + 1 # we signal the next coming batch -> where work should be continued from
    checkpoint_data = struct.pack('!ii', epoch, checkpointed_idx)
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
