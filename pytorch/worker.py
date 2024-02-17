import asyncio
import os
import torch

from pathlib import Path
from torch import nn

from pytorch.constants import (
    WORKER_DIR,
    DATA_PATH,
    GRADIENT_FILE,
    GRADIENT_READY_FILE,
    WORKER_FINISHED_FILE,
    STATE_DICT_FILE,
    STATE_DICT_READY_FILE,
    WAITING_PERIOD,
    MONITORING_PERIOD,
    MONITOR_FILE
)
from pytorch.logger import logger
from pytorch.utils import load_network, load_optimizer, user_script


LOG_INTERVAL = 50


class Worker:
    def __init__(self):
        self.monitor_path = self.get_path(MONITOR_FILE)
        self.gradient_path = self.get_path(GRADIENT_FILE)
        self.gradient_ready_path = self.get_path(GRADIENT_READY_FILE)
        self.worker_finished_path = self.get_path(WORKER_FINISHED_FILE)
        self.state_dict_path = self.get_path(STATE_DICT_FILE)
        self.state_dict_ready_path = self.get_path(STATE_DICT_READY_FILE)

    @staticmethod
    def get_path(file_or_directory: str) -> Path:
        return WORKER_DIR / file_or_directory

    @staticmethod
    async def wait_data():
        while not os.path.exists(DATA_PATH):
            await asyncio.sleep(WAITING_PERIOD)
        logger.info("Data has arrived.")

    @staticmethod
    def is_last_iteration(epoch: int, batch_idx: int, total_batches: int) -> bool:
        return epoch == user_script.N_EPOCHS - 1 and batch_idx == total_batches - 1

    def signal_worker_finished(self):
        with open(self.worker_finished_path, "wb"):
            pass

    async def wait_state_dict(self):
        while not os.path.exists(self.state_dict_ready_path):
            await asyncio.sleep(WAITING_PERIOD)
        if not os.path.exists(self.state_dict_path):
            raise FileNotFoundError(f"{self.state_dict_path} does not exist!")
        os.remove(self.state_dict_ready_path)
        logger.debug("state dict waited")

    def save_gradient(self, network: nn.Module):
        gradient = {name: param.grad.data for name, param in network.named_parameters()}
        torch.save(gradient, self.gradient_path)

        with open(self.gradient_ready_path, "wb"):
            pass

    async def monitor(self, task: asyncio.Task):
        logger.info("Worker monitor started.")
        while not task.done():
            with open(self.monitor_path, "wb"):
                pass
            await asyncio.sleep(MONITORING_PERIOD)

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

                if self.is_last_iteration(epoch, batch_idx, total_batches):
                    self.signal_worker_finished()
                self.save_gradient(network)

                await self.wait_state_dict()

                if batch_idx % LOG_INTERVAL == 0:
                    logger.info(f"Worker: Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.6f}")

        logger.info("Worker finished.")

    async def run(self):
        train_network_task = asyncio.create_task(self.train_network())
        monitor_task = asyncio.create_task(self.monitor(train_network_task))
        await asyncio.gather(train_network_task, monitor_task)
