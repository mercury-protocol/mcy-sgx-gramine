import asyncio
import os
import torch

from pathlib import Path
from torch import nn

from user_script import train_batch, data_loader_factory, N_EPOCHS

from pytorch.constants import (
    WORKER_DIR,
    DATA_DIR,
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


class Worker:
    def __init__(self, node: str):
        self.node = node
        self.monitor_path = self.get_path(MONITOR_FILE)
        self.data_path = self.get_path(DATA_DIR)
        self.optimizer_path = self.get_path(OPTIMIZER_FILE)
        self.gradient_path = self.get_path(GRADIENT_FILE)
        self.training_complete_path = self.get_path(TRAINING_COMPLETE_FILE)
        self.state_dict_path = self.get_path(STATE_DICT_FILE)

    def get_path(self, file_or_directory: str) -> Path:
        return WORKER_DIR / self.node / file_or_directory

    async def wait_data(self):
        while not os.path.exists(self.data_path):
            await asyncio.sleep(WAITING_PERIOD)

    @staticmethod
    def is_training_complete(epoch: int, batch_idx: int, total_batches: int) -> bool:
        return epoch == N_EPOCHS - 1 and batch_idx == total_batches - 1

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
            print(f"Worker {self.node} monitor: Worker {self.node} is running.")
            await asyncio.sleep(MONITOR_PERIOD)

        print(f"Worker {self.node} monitor finished.")
        return

    async def train_network(self):
        print(f"Worker {self.node} started.")
        await self.wait_data()
        data_loader = data_loader_factory.create(self.data_path)
        total_batches = len(data_loader)

        for epoch in range(N_EPOCHS):
            for batch_idx, (data, target) in enumerate(data_loader):
                network = load_network(path=self.state_dict_path, delete_file=True)
                optimizer = load_optimizer(network, path=self.optimizer_path)
                loss = train_batch(data, target, network, optimizer)

                if self.is_training_complete(epoch, batch_idx, total_batches):
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
