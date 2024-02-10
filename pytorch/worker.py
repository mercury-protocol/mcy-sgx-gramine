import asyncio
import os
import torch

from pathlib import Path
from torch import nn

from pytorch.constants import (
    WORKER_DIR,
    DATA_PATH,
    GRADIENT_FILE,
    TRAINING_COMPLETE_FILE,
    STATE_DICT_FILE,
    WAITING_PERIOD,
    MONITOR_PERIOD,
    MONITOR_FILE
)
from pytorch.logger import logger
from pytorch.utils import load_network, load_optimizer, user_script


LOG_INTERVAL = 50


class Worker:
    def __init__(self):
        self.monitor_path = self.get_path(MONITOR_FILE)
        self.gradient_path = self.get_path(GRADIENT_FILE)
        self.training_complete_path = self.get_path(TRAINING_COMPLETE_FILE)
        self.state_dict_path = self.get_path(STATE_DICT_FILE)

    @staticmethod
    def get_path(file_or_directory: str) -> Path:
        return WORKER_DIR / file_or_directory

    @staticmethod
    async def wait_data():
        while not os.path.exists(DATA_PATH):
            await asyncio.sleep(WAITING_PERIOD)
        logger.info("Worker: data has arrived.")

    @staticmethod
    def is_training_complete(epoch: int, batch_idx: int, total_batches: int) -> bool:
        return epoch == user_script.N_EPOCHS - 1 and batch_idx == total_batches - 1

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
        logger.info("Worker monitor started.")
        while not task.done():
            with open(self.monitor_path, "wb"):
                pass
            logger.info("Worker monitor: Worker is running.")
            await asyncio.sleep(MONITOR_PERIOD)

        logger.info("Worker monitor finished.")
        return

    async def train_network(self):
        logger.info("Worker started.")
        await self.wait_data()
        data_loader = user_script.data_loader_factory.create(DATA_PATH)
        total_batches = len(data_loader)

        for epoch in range(user_script.N_EPOCHS):
            for batch_idx, (data, target) in enumerate(data_loader):
                network = load_network(path=self.state_dict_path, delete_file=True)
                optimizer = load_optimizer(network)
                loss = user_script.train_batch(data, target, network, optimizer)

                if self.is_training_complete(epoch, batch_idx, total_batches):
                    self.signal_training_complete()
                self.save_gradient(network)
                await self.wait_network_aggregation()

                if batch_idx % LOG_INTERVAL == 0:
                    logger.info(f"Worker: Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.6f}")

        logger.info("Worker finished.")

    async def run(self):
        train_network_task = asyncio.create_task(self.train_network())
        monitor_task = asyncio.create_task(self.monitor(train_network_task))
        await asyncio.gather(train_network_task, monitor_task)
