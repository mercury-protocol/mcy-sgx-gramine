import asyncio
import os
import torch

from pathlib import Path
from torch import nn

from mcy_dist_ai.constants import (
    BASE_DIR,
    STATE_DICT_PATH,
    WORKER_FINISHED_FILE,
    STATE_DICT_READY_PATH,
    GRADIENT_FILE,
    GRADIENT_READY_FILE,
    WAITING_PERIOD,
    MONITOR_PATH,
    MONITORING_PERIOD,
    LOG_INTERVAL,
    TRAINED_MODEL_PATH
)
from mcy_dist_ai.logger import logger
from mcy_dist_ai.utils import torch_safe_load, load_model, load_optimizer, list_worker_nodes


class Leader:
    def __init__(self):
        self.worker_finished_paths = list()
        self.gradient_paths = list()
        self.gradient_ready_paths = list()
        for node in list_worker_nodes():
            self.worker_finished_paths.append(self.get_path(node, WORKER_FINISHED_FILE))
            self.gradient_paths.append(self.get_path(node, GRADIENT_FILE))
            self.gradient_ready_paths.append(self.get_path(node, GRADIENT_READY_FILE))

    @staticmethod
    def get_path(worker_node: str, file: str) -> Path:
        if "." in file:
            name, extension = file.split(".")
            file = f"{name}_{worker_node}.{extension}"
        else:
            file = f"{file}_{worker_node}"
        return BASE_DIR / file

    def have_workers_finished(self) -> bool:
        return all(os.path.exists(path) for path in self.worker_finished_paths)

    async def wait_gradients(self):
        while not all(os.path.exists(path) for path in self.gradient_ready_paths):
            await asyncio.sleep(WAITING_PERIOD)
        if not all(os.path.exists(path) for path in self.gradient_paths):
            raise FileNotFoundError("Not all gradient files exist!")
        # TODO: locking mechanism should be used here,
        #  now we just sleep a little to give time the other process to finish copying
        await asyncio.sleep(WAITING_PERIOD)
        [os.remove(path) for path in self.gradient_ready_paths]
        logger.debug("gradients waited")

    def delete_gradients(self):
        for path in self.gradient_paths:
            if os.path.exists(path):
                os.remove(path)
        logger.debug("gradients deleted")

    @staticmethod
    def save_state_dict(model: nn.Module):
        torch.save(model.state_dict(), STATE_DICT_PATH)
        with open(STATE_DICT_READY_PATH, "wb"):
            pass

    @staticmethod
    def save_trained_model(model: nn.Module):
        torch.save(model.state_dict(), TRAINED_MODEL_PATH)

    def aggregate_gradients(self, model: nn.Module):
        gradients = [torch_safe_load(path) for path in self.gradient_paths]

        for i, param in enumerate(model.parameters()):
            param.grad = torch.mean(
                torch.stack(
                    [grad[i] for grad in gradients]
                ),
                dim=0
            )

        logger.debug("gradients aggregated")

    @staticmethod
    async def monitor(task: asyncio.Task):
        logger.info("Monitor started.")
        while not task.done():
            with open(MONITOR_PATH, "wb"):
                pass
            await asyncio.sleep(MONITORING_PERIOD)

        logger.info("Monitor finished.")

    async def aggregate_model(self):
        logger.info("Leader started.")
        model = load_model()
        optimizer = load_optimizer(model)

        aggr_idx = 0
        while True:
            await self.wait_gradients()
            self.aggregate_gradients(model)

            optimizer.step()
            optimizer.zero_grad()

            self.save_state_dict(model)

            self.delete_gradients()

            if aggr_idx % LOG_INTERVAL == 0:
                logger.info(f"{aggr_idx}th aggregation completed.")
            aggr_idx += 1

            if self.have_workers_finished():
                self.save_trained_model(model)
                logger.info("Leader finished.")
                return

    async def run(self):
        aggregate_model_task = asyncio.create_task(self.aggregate_model())
        monitor_task = asyncio.create_task(self.monitor(aggregate_model_task))
        await asyncio.gather(aggregate_model_task, monitor_task)
