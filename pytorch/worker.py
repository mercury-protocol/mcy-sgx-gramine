import asyncio
import os
import torch

from pathlib import Path
from torch import nn

from user_script import train_batch, data_loader_factory, N_EPOCHS

from pytorch.constants import (
    WORKER_DIR,
    SPLIT_DATA_PATH,
    OPTIMIZER_FILE,
    GRADIENT_FILE,
    TRAINING_COMPLETE_FILE,
    STATE_DICT_FILE,
    WAITING_PERIOD,
    MONITOR_PERIOD,
    MONITOR_FILE
)
from pytorch.utils import load_network, load_optimizer


LOG_INTERVAL = 50


def get_path(node: str, file: str) -> Path:
    return WORKER_DIR / node / file


class Worker:
    def __init__(self, node: str):
        self.node = node
        self.monitor_path = get_path(node, MONITOR_FILE)
        self.data_path = SPLIT_DATA_PATH / node
        self.optimizer_path = get_path(node, OPTIMIZER_FILE)
        self.gradient_path = get_path(node, GRADIENT_FILE)
        self.training_complete_path = get_path(node, TRAINING_COMPLETE_FILE)
        self.state_dict_path = get_path(node, STATE_DICT_FILE)
        self.data_loader = data_loader_factory.create(self.data_path)
        self.total_batches = len(self.data_loader)

    def is_training_complete(self, epoch: int, batch_idx: int) -> bool:
        return epoch == N_EPOCHS - 1 and batch_idx == self.total_batches - 1

    def signal_training_complete(self):
        with open(self.training_complete_path, "wb"):
            pass

    async def wait_network_aggregation(self):
        while not os.path.exists(self.state_dict_path):
            await asyncio.sleep(WAITING_PERIOD)

    def save_gradient(self, network: nn.Module):
        gradient = {name: param.grad.data for name, param in network.named_parameters()}
        torch.save(gradient, self.gradient_path)

    async def monitor(self, task: asyncio.Task):
        print(f"Worker {self.node} monitor started.")
        while not task.done():
            with open(self.monitor_path, "wb"):
                pass
            await asyncio.sleep(MONITOR_PERIOD)

        print(f"Worker {self.node} monitor finished.")
        return

    async def train_network(self):
        print(f"Worker {self.node} started.")

        for epoch in range(N_EPOCHS):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                network = load_network(path=self.state_dict_path, delete_file=True)
                optimizer = load_optimizer(network, path=self.optimizer_path)
                loss = train_batch(data, target, network, optimizer)

                if self.is_training_complete(epoch, batch_idx):
                    self.signal_training_complete()
                self.save_gradient(network)
                await self.wait_network_aggregation()

                if batch_idx % LOG_INTERVAL == 0:
                    print(f"Worker: {self.node} Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.6f}")

        print(f"Worker {self.node} finished.")

    async def run(self):
        train_network_task = asyncio.create_task(self.train_network())
        monitor_task = asyncio.create_task(self.monitor(train_network_task))
        await asyncio.gather(train_network_task, monitor_task)
