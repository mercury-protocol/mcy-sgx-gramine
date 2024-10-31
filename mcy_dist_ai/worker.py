import asyncio
import os
import torch

from pathlib import Path
from torch import nn

from mcy_dist_ai.args import TENSOR_LOAD, WORKER_NODES_NUM
from mcy_dist_ai.constants import (
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
    TRAINED_MODEL_PATH,
)
from mcy_dist_ai.logger import logger
from mcy_dist_ai.utils import (
    create_tensor_loader,
    load_model,
    load_optimizer,
    user_script,
    checkpoint,
    load_last_checkpoint,
    safe_create_extra_training_args,
)

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
    def create_data_loader():
        if TENSOR_LOAD:
            return create_tensor_loader(str(DATA_PATH))
        else:
            return user_script.create_data_loader(str(DATA_PATH))

    @staticmethod
    def is_last_iteration(epoch: int, batch_idx: int, total_batches: int) -> bool:
        return epoch == user_script.N_EPOCHS - 1 and batch_idx == total_batches - 1

    def signal_worker_finished(self):
        with open(self.worker_finished_path, "wb"):
            pass

    @staticmethod
    def save_trained_model(model: nn.Module):
        torch.save(model.state_dict(), TRAINED_MODEL_PATH)

    @staticmethod
    async def wait_state_dict():
        while not os.path.exists(STATE_DICT_READY_PATH):
            await asyncio.sleep(WAITING_PERIOD)
        if not os.path.exists(STATE_DICT_PATH):
            raise FileNotFoundError(f"{STATE_DICT_PATH} does not exist!")
        # TODO: locking mechanism should be used here,
        #  now we just sleep a little to give time the other process to finish copying
        await asyncio.sleep(WAITING_PERIOD)
        os.remove(STATE_DICT_READY_PATH)
        logger.debug("state dict waited")

    def save_gradient(self, model: nn.Module):
        gradient = [param.grad.data for param in model.parameters()]
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

    async def train_model(self):
        logger.info("Worker started.")
        await self.wait_data()

        data_loader = self.create_data_loader()
        total_batches = len(data_loader)
        model = load_model(path=STATE_DICT_PATH, delete_file=True)
        optimizer = load_optimizer(model)
        extra_args = safe_create_extra_training_args(data_loader, optimizer)

        # TODO: this is probably needed because recovery - investigate why
        if os.path.exists(STATE_DICT_READY_PATH):
            os.remove(STATE_DICT_READY_PATH)

        start_epoch, start_batch = load_last_checkpoint()
        for epoch in range(start_epoch, user_script.N_EPOCHS):
            for batch_idx, batch in enumerate(data_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue

                loss = user_script.train_batch(batch, model, optimizer, *extra_args)
                
                self.save_gradient(model)
                checkpoint(epoch=epoch, batch_idx=batch_idx)

                # TODO: If moved above save_gradient, leader fails to send confirmation to watcher
                #  - probably a bug in Vulkan
                if self.is_last_iteration(epoch, batch_idx, total_batches):
                    self.signal_worker_finished()
                    self.save_trained_model(model)
                elif WORKER_NODES_NUM != 1:
                    # if there's only 1 worker, leader is not needed
                    await self.wait_state_dict()
                    model = load_model(path=STATE_DICT_PATH, delete_file=True)
                    optimizer = load_optimizer(model)

                if batch_idx % LOG_INTERVAL == 0:
                    logger.info(f"Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.6f}")

        logger.info("Worker finished.")

    async def run(self):
        training_task = asyncio.create_task(self.train_model())
        monitor_task = asyncio.create_task(self.monitor(training_task))
        await asyncio.gather(training_task, monitor_task)
