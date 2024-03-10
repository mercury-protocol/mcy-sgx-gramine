import asyncio
import os
import torch

from pathlib import Path
from torch import nn

from pytorch.constants import (
    ROLE,
    WORKER_LLM_ROLE,
    BASE_DIR,
    DATA_PATH,
    GRADIENT_FILE,
    GRADIENT_READY_FILE,
    WORKER_FINISHED_FILE,
    STATE_DICT_PATH,
    STATE_DICT_READY_PATH,
    WAITING_PERIOD,
    MONITORING_PERIOD,
    MONITOR_PATH,
    LOG_INTERVAL,
    WORKER_NODES_NUM,
    TRAINED_MODEL_PATH,
)
from pytorch.logger import logger
from pytorch.utils import load_network, load_optimizer, user_script, checkpoint, load_last_checkpoint


class Worker:
    def __init__(self):
        self.gradient_path = self.get_path(GRADIENT_FILE)
        self.gradient_ready_path = self.get_path(GRADIENT_READY_FILE)
        self.worker_finished_path = self.get_path(WORKER_FINISHED_FILE)

    @staticmethod
    def get_path(file_or_directory: str) -> Path:
        return BASE_DIR / file_or_directory

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

    @staticmethod
    def save_trained_model(network: nn.Module):
        torch.save(network.state_dict(), TRAINED_MODEL_PATH)

    @staticmethod
    async def wait_state_dict():
        if WORKER_NODES_NUM == 1:
            # if there's only 1 worker, leader is not needed
            return
        while not os.path.exists(STATE_DICT_READY_PATH):
            await asyncio.sleep(WAITING_PERIOD)
        if not os.path.exists(STATE_DICT_PATH):
            raise FileNotFoundError(f"{STATE_DICT_PATH} does not exist!")
        os.remove(STATE_DICT_READY_PATH)
        logger.debug("state dict waited")

    def save_gradient(self, network: nn.Module):
        gradient = {name: param.grad.data for name, param in network.named_parameters()}
        torch.save(gradient, self.gradient_path)

        with open(self.gradient_ready_path, "wb"):
            pass

    @staticmethod
    async def monitor(task: asyncio.Task):
        logger.info("Monitor started.")
        while not task.done():
            with open(MONITOR_PATH, "wb"):
                pass
            await asyncio.sleep(MONITORING_PERIOD)

        logger.info("Monitor finished.")
        return

    async def train_network(self):
        logger.info("Worker started.")
        await self.wait_data()

        data_loader = user_script.data_loader_factory.create(DATA_PATH)
        total_batches = len(data_loader)

        # TODO: this is probably needed because recovery - investigate why
        if os.path.exists(STATE_DICT_READY_PATH):
            os.remove(STATE_DICT_READY_PATH)

        start_epoch, start_batch = load_last_checkpoint()
        for epoch in range(start_epoch, user_script.N_EPOCHS):
            for batch_idx, (data, target) in enumerate(data_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue

                network = load_network(path=STATE_DICT_PATH, delete_file=True)
                optimizer = load_optimizer(network)
                loss = user_script.train_batch(data, target, network, optimizer)
                
                self.save_gradient(network)
                checkpoint(epoch=epoch, batch_idx=batch_idx)

                # TODO: If moved above save_gradient, leader fails to send confirmation to watcher
                # probably a bug in Vulkan
                if self.is_last_iteration(epoch, batch_idx, total_batches):
                    self.signal_worker_finished()
                    self.save_trained_model(network)
                else:
                    await self.wait_state_dict()

                if batch_idx % LOG_INTERVAL == 0:
                    logger.info(f"Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.6f}")

        logger.info("Worker finished.")

    @staticmethod
    async def fine_tune_llm():
        # TODO: make this implementation compatible with the original code flow:
        # - distributed training
        # - use Mercury user script format
        # - no separate role for llm training
        logger.info("Worker started - Fine tune LLM")
        user_script.main()

    async def run(self):
        training_task = asyncio.create_task(
            self.fine_tune_llm() if ROLE == WORKER_LLM_ROLE else self.train_network()
        )
        monitor_task = asyncio.create_task(self.monitor(training_task))
        await asyncio.gather(training_task, monitor_task)
