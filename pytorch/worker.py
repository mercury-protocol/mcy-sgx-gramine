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


class Worker:
    def __init__(self, node: str):
        self.node = node
        self.data_path = SPLIT_DATA_PATH / node
        self.optimizer_path = get_file_path(node, FileEnum.OPTIMIZER)
        self.gradient_path = get_file_path(node, FileEnum.GRADIENT)
        self.training_complete_path = get_file_path(node, FileEnum.TRAINING_COMPLETE)
        self.batch_training_complete_path = get_file_path(node, FileEnum.BATCH_TRAINING_COMPLETE)
        self.batch_aggregation_complete_path = get_file_path(node, FileEnum.BATCH_AGGREGATION_COMPLETE)
        self.data_loader = data_loader_factory.create(self.data_path)
        self.total_batches = len(self.data_loader)

    def is_training_complete(self, epoch: int, batch_idx: int) -> bool:
        return epoch == N_EPOCHS - 1 and batch_idx == self.total_batches - 1

    def signal_training_complete(self):
        with open(self.training_complete_path, "wb"):
            pass

    def signal_batch_training_complete(self):
        with open(self.batch_training_complete_path, "wb"):
            pass

    async def wait_batch_aggregation(self):
        path = self.batch_aggregation_complete_path
        while not os.path.exists(path):
            await asyncio.sleep(WAITING_PERIOD)
        safe_delete_file(path)

    def save_gradients(self, network: nn.Module):
        gradients = {name: param.grad.data for name, param in network.named_parameters()}
        torch.save(gradients, self.gradient_path)

    async def train_network(self):
        print(f"Worker {self.node} started its training.")

        for epoch in range(N_EPOCHS):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                network = load_network(path=AGGREGATED_STATE_DICT_PATH)
                optimizer = load_optimizer(network, path=self.optimizer_path)
                loss = train_batch(data, target, network, optimizer)
                self.save_gradients(network)

                if self.is_training_complete(epoch, batch_idx):
                    self.signal_training_complete()
                self.signal_batch_training_complete()
                await self.wait_batch_aggregation()

                if batch_idx % LOG_INTERVAL == 0:
                    print(f"Worker: {self.node} Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.6f}")

        print(f"Worker {self.node} finished its training.")
