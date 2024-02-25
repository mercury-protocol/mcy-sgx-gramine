import importlib
import os
import torch
import argparse
import sys
import struct

from pathlib import Path
from time import sleep
from torch import nn
from typing import Any, List, Union

import pytorch.constants as constants
from pytorch.constants import WAITING_PERIOD, WORKER_ROLE, LEADER_ROLE, STATE_DICT_READY_FILE
from pytorch.logger import logger

parser = argparse.ArgumentParser()
parser.add_argument("--role", type=str, help="Node role - leader or worker")
parser.add_argument("--worker_count", type=int, help="Worker nodes count")
args = parser.parse_args()
if args.role is None:
    print("Role argument is missing")
    sys.exit(1)
elif args.role != WORKER_ROLE and args.role != LEADER_ROLE:
    print(f"role must be {WORKER_ROLE} or {LEADER_ROLE}")
    sys.exit(1)

if args.role == LEADER_ROLE and args.worker_count is None:
    print("Worker nodes count argument is required for leader")
    sys.exit(1)

constants.setup(role=args.role, worker_nodes_num=args.worker_count)

script_path = os.path.abspath(constants.USER_SCRIPT_PATH)
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

    logger.info("network loaded")
    return network


def load_optimizer(network: nn.Module) -> Any:
    optimizer = user_script.optimizer_factory.create(network.parameters())
    return optimizer


def list_worker_nodes() -> List[str]:
    return [str(i + 1) for i in range(constants.WORKER_NODES_NUM)]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str, help="Node role - leader or worker")
    parser.add_argument("--worker_count", type=int, help="Worker nodes count")
    args = parser.parse_args()
    if args.role is None:
        print("Role argument is missing")
        sys.exit(1)
    elif args.role != WORKER_ROLE or args.role != LEADER_ROLE:
        print(f"role must be {WORKER_ROLE} or {LEADER_ROLE}")
        sys.exit(1)

    if args.role == LEADER_ROLE and args.worker_count is None:
        print("Worker nodes count argument is required for leader")
        sys.exit(1)
    
    constants.setup(role=args.role, worker_nodes_num=args.worker_count)
    return args

def checkpoint(epoch, batch):
    fname = f"checkpoint.bin"
    checkpoint_data = struct.pack('!ii', epoch, batch)
    with open(fname, 'wb') as f:
        f.write(checkpoint_data)

def load_last_checkpoint():
     fname = f"checkpoint.bin"
     if os.path.exists(fname) == False:
          print("checkpoint file does not exist")
          return 0, 0
     
     with open(fname, 'rb') as f:
          checkpoint_data = f.read()
     epoch, batch = struct.unpack('!ii', checkpoint_data)

     return epoch, batch