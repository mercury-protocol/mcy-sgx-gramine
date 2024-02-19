import asyncio
import os
import torch

from pathlib import Path
from torch import nn

from pytorch.constants import (
    LEADER_DIR,
    AGGREGATED_STATE_DICT_PATH,
    WORKER_FINISHED_FILE,
    STATE_DICT_READY_FILE,
    GRADIENT_FILE,
    GRADIENT_READY_FILE,
    WAITING_PERIOD,
    MONITOR_FILE,
    MONITORING_PERIOD,
    LOG_INTERVAL
)
from pytorch.logger import logger
from pytorch.utils import torch_safe_load, load_network, load_optimizer, list_worker_nodes


class Leader:
    def __init__(self):
        self.worker_finished_paths = list()
        self.gradient_paths = list()
        self.gradient_ready_paths = list()
        self.state_dict_ready_path = LEADER_DIR / STATE_DICT_READY_FILE
        self.monitor_path = LEADER_DIR / MONITOR_FILE
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
        return LEADER_DIR / file

    def have_workers_finished(self) -> bool:
        return all(os.path.exists(path) for path in self.worker_finished_paths)

    async def wait_gradients(self):
        while not all(os.path.exists(path) for path in self.gradient_ready_paths):
            await asyncio.sleep(WAITING_PERIOD)
        if not all(os.path.exists(path) for path in self.gradient_paths):
            raise FileNotFoundError("Not all gradient files exist!")
        [os.remove(path) for path in self.gradient_ready_paths]
        logger.debug("gradients waited")

    def delete_gradients(self):
        for path in self.gradient_paths:
            if os.path.exists(path):
                os.remove(path)
        logger.debug("gradients deleted")

    def save_state_dict(self, network: nn.Module):
        torch.save(network.state_dict(), AGGREGATED_STATE_DICT_PATH)
        with open(self.state_dict_ready_path, "wb"):
            pass

    def aggregate_gradients(self, network: nn.Module):
        gradients = [torch_safe_load(path) for path in self.gradient_paths]
        avg_grads = gradients[0]
        num = len(gradients)

        for grad in gradients[1:]:
            for name, _ in network.named_parameters():
                avg_grads[name] = torch.add(avg_grads[name], grad[name])

        for name, param in network.named_parameters():
            param.grad = avg_grads[name] / num

        logger.debug("gradients aggregated")

    async def monitor(self, task: asyncio.Task):
        logger.info("Monitor started.")
        while not task.done():
            with open(self.monitor_path, "wb"):
                pass
            await asyncio.sleep(MONITORING_PERIOD)

        logger.info("Monitor finished.")

    async def aggregate_network(self):
        logger.info("Leader started.")
        network = load_network()

        aggr_idx = 0
        while True:
            await self.wait_gradients()
            self.aggregate_gradients(network)

            optimizer = load_optimizer(network)
            optimizer.step()
            optimizer.zero_grad()

            self.save_state_dict(network)

            self.delete_gradients()

            if aggr_idx % LOG_INTERVAL == 0:
                logger.info(f"{aggr_idx}th aggregation completed.")
            aggr_idx += 1

            if self.have_workers_finished():
                logger.info("Leader finished.")
                return

    async def run(self):
        aggregate_network_task = asyncio.create_task(self.aggregate_network())
        monitor_task = asyncio.create_task(self.monitor(aggregate_network_task))
        await asyncio.gather(aggregate_network_task, monitor_task)
