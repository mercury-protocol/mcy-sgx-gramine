import asyncio
import os
import torch

from pathlib import Path
from torch import nn
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer

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
from pytorch.utils import load_model, load_optimizer, user_script, checkpoint, load_last_checkpoint


class VulkanCallback(TrainerCallback):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def save_gradients(self):
        gradient = {name: param.data for name, param in self.trainer.model.named_parameters() if param.requires_grad}
        torch.save(gradient, BASE_DIR / GRADIENT_FILE)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # self.trainer.model = load_model(path=STATE_DICT_PATH, delete_file=True)
        # optimizer = load_optimizer(model)
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.save_gradients()
        # checkpoint(epoch=state.epoch, batch_idx=batch_idx)

        # if state.global_step == state.max_steps:
        #     self.signal_worker_finished()
        # else:
        #     await self.wait_state_dict()
        print(state.epoch, state.global_step, state.max_steps)


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
    def save_trained_model(model: nn.Module):
        torch.save(model.state_dict(), TRAINED_MODEL_PATH)

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

    def save_gradient(self, model: nn.Module):
        gradient = {name: param.grad.data for name, param in model.named_parameters()}
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

        data_loader = user_script.create_data_loader(DATA_PATH)
        total_batches = len(data_loader)

        # TODO: this is probably needed because recovery - investigate why
        if os.path.exists(STATE_DICT_READY_PATH):
            os.remove(STATE_DICT_READY_PATH)

        start_epoch, start_batch = load_last_checkpoint()
        for epoch in range(start_epoch, user_script.N_EPOCHS):
            for batch_idx, (data, target) in enumerate(data_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue

                model = load_model(path=STATE_DICT_PATH, delete_file=True)
                optimizer = load_optimizer(model)
                loss = user_script.train_batch(data, target, model, optimizer)
                
                self.save_gradient(model)
                checkpoint(epoch=epoch, batch_idx=batch_idx)

                # TODO: If moved above save_gradient, leader fails to send confirmation to watcher
                # probably a bug in Vulkan
                if self.is_last_iteration(epoch, batch_idx, total_batches):
                    self.signal_worker_finished()
                    self.save_trained_model(model)
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

        trainer = user_script.trainer
        callback = VulkanCallback(trainer=trainer)
        trainer.add_callback(callback)
        trainer.train()

    async def run(self):
        training_task = asyncio.create_task(
            self.fine_tune_llm() if ROLE == WORKER_LLM_ROLE else self.train_model()
        )
        monitor_task = asyncio.create_task(self.monitor(training_task))
        await asyncio.gather(training_task, monitor_task)
