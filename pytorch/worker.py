import asyncio
import os
import torch
from torch import nn

from user_script import train_batch, data_loader_factory, N_EPOCHS

from pytorch.constants import (
    SPLIT_DATA_PATH,
    OPTIMIZER_FILE,
    GRADIENT_FILE,
    TRAINING_COMPLETE_FILE,
    STATE_DICT_FILE,
    WAITING_PERIOD
)
from pytorch.utils import load_network, load_optimizer, get_file_path


LOG_INTERVAL = 10


class Worker:
    def __init__(self, node: str):
        self.node = node
        self.data_path = SPLIT_DATA_PATH / node
        self.optimizer_path = get_file_path(node, OPTIMIZER_FILE)
        self.gradient_path = get_file_path(node, GRADIENT_FILE)
        self.training_complete_path = get_file_path(node, TRAINING_COMPLETE_FILE)
        self.state_dict_path = get_file_path(node, STATE_DICT_FILE)
        self.data_loader = data_loader_factory.create(self.data_path)
        self.total_batches = len(self.data_loader)

    def is_training_complete(self, epoch: int, batch_idx: int) -> bool:
        return epoch == N_EPOCHS - 1 and batch_idx == self.total_batches - 1

    def signal_training_complete_to_leader(self):
        with open(self.training_complete_path, "wb"):
            pass

    async def wait_leader_network_aggregation(self):
        while not os.path.exists(self.state_dict_path):
            await asyncio.sleep(WAITING_PERIOD)

    def send_gradients_to_leader(self, network: nn.Module):
        gradients = {name: param.grad.data for name, param in network.named_parameters()}
        torch.save(gradients, self.gradient_path)

    async def train_network(self):
        print(f"Worker {self.node} started its training.")

        for epoch in range(N_EPOCHS):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                network = load_network(path=self.state_dict_path, delete_file=True)
                optimizer = load_optimizer(network, path=self.optimizer_path)
                loss = train_batch(data, target, network, optimizer)

                if self.is_training_complete(epoch, batch_idx):
                    self.signal_training_complete_to_leader()
                self.send_gradients_to_leader(network)
                await self.wait_leader_network_aggregation()

                if batch_idx % LOG_INTERVAL == 0:
                    print(f"Worker: {self.node} Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.6f}")

        print(f"Worker {self.node} finished its training.")
