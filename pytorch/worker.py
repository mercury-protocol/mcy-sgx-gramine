import asyncio
import os
import torch

from user_script import train_batch, data_loader_factory, N_EPOCHS

from pytorch.constants import AGGREGATED_STATE_DICT_PATH, WAITING_PERIOD
from pytorch.utils import (
    load_network, load_optimizer, save_gradients, get_file_path, get_data_path, DirEnum, FileEnum, safe_delete_file
)


LOG_INTERVAL = 10
train_losses = []
train_counter = []


async def train_network(node: str):
    print(f"Worker {node} started its training.")
    data_path = get_data_path(node)
    state_dict_path = get_file_path(DirEnum.WORKER, node, FileEnum.STATE_DICT)
    optimizer_path = get_file_path(DirEnum.WORKER, node, FileEnum.OPTIMIZER)
    gradient_path = get_file_path(DirEnum.WORKER, node, FileEnum.GRADIENT)
    batch_aggregation_complete_path = get_file_path(DirEnum.WORKER, node, FileEnum.BATCH_AGGREGATION_COMPLETE)
    batch_training_complete_path = get_file_path(DirEnum.WORKER, node, FileEnum.BATCH_TRAINING_COMPLETE)
    training_complete_path = get_file_path(DirEnum.WORKER, node, FileEnum.TRAINING_COMPLETE)

    data_loader = data_loader_factory.create(data_path)
    total_batches = len(data_loader)

    for epoch in range(N_EPOCHS):
        for batch_idx, (data, target) in enumerate(data_loader):
            safe_delete_file(batch_aggregation_complete_path)

            network = load_network(path=AGGREGATED_STATE_DICT_PATH)
            optimizer = load_optimizer(network, path=optimizer_path)
            loss = train_batch(data, target, network, optimizer)
            torch.save(network.state_dict(), state_dict_path)
            save_gradients(network, gradient_path)

            if epoch == N_EPOCHS - 1 and batch_idx == total_batches - 1:
                # create a file to signal that training is complete
                with open(training_complete_path, "wb"):
                    pass

            # create a file to signal that batch training is complete
            with open(batch_training_complete_path, "wb"):
                pass

            # wait other workers to finish and leader to aggregate
            while not os.path.exists(batch_aggregation_complete_path):
                await asyncio.sleep(WAITING_PERIOD)

            if batch_idx % LOG_INTERVAL == 0:
                print(f"Worker: {node} Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.6f}")

    print(f"Worker {node} finished its training.")
