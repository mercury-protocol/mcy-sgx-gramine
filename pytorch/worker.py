import asyncio
import os
import torch
from torch import nn

from user_script import train_batch, data_loader_factory, N_EPOCHS

from pytorch.constants import AGGREGATED_STATE_DICT_PATH, SPLIT_DATA_PATH, WAITING_PERIOD
from pytorch.utils import (
    load_network, load_optimizer, get_file_path, FileEnum, safe_delete_file
)


LOG_INTERVAL = 10


def is_training_complete(epoch: int, batch_idx: int, total_batches: int) -> bool:
    return epoch == N_EPOCHS - 1 and batch_idx == total_batches - 1


def signal_training_complete(node: str):
    with open(get_file_path(node, FileEnum.TRAINING_COMPLETE), "wb"):
        pass


def signal_batch_training_complete(node: str):
    with open(get_file_path(node, FileEnum.BATCH_TRAINING_COMPLETE), "wb"):
        pass


async def wait_batch_aggregation(node: str):
    path = get_file_path(node, FileEnum.BATCH_AGGREGATION_COMPLETE)
    while not os.path.exists(path):
        await asyncio.sleep(WAITING_PERIOD)
    safe_delete_file(path)


def save_gradients(network: nn.Module, path):
    gradients = {name: param.grad.data for name, param in network.named_parameters()}
    torch.save(gradients, path)


async def train_network(node: str):
    print(f"Worker {node} started its training.")
    data_path = SPLIT_DATA_PATH / node
    optimizer_path = get_file_path(node, FileEnum.OPTIMIZER)
    gradient_path = get_file_path(node, FileEnum.GRADIENT)

    data_loader = data_loader_factory.create(data_path)
    total_batches = len(data_loader)

    for epoch in range(N_EPOCHS):
        for batch_idx, (data, target) in enumerate(data_loader):
            network = load_network(path=AGGREGATED_STATE_DICT_PATH)
            optimizer = load_optimizer(network, path=optimizer_path)
            loss = train_batch(data, target, network, optimizer)
            save_gradients(network, gradient_path)

            if is_training_complete(epoch, batch_idx, total_batches):
                signal_training_complete(node)
            signal_batch_training_complete(node)
            await wait_batch_aggregation(node)

            if batch_idx % LOG_INTERVAL == 0:
                print(f"Worker: {node} Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.6f}")

    print(f"Worker {node} finished its training.")
