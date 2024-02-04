import asyncio
import os
import torch

from pathlib import Path
from torch import nn

from pytorch.constants import (
    LEADER_DIR,
    AGGREGATED_STATE_DICT_PATH,
    TRAINING_COMPLETE_FILE,
    BATCH_AGGREGATION_COMPLETE_FILE,
    GRADIENT_FILE,
    WAITING_PERIOD,
    MONITOR_FILE,
    MONITOR_PERIOD
)
from pytorch.utils import load_network, load_optimizer, list_worker_nodes


class Leader:
    def __init__(self):
        self.training_complete_paths = list()
        self.gradient_paths = list()
        self.batch_aggregation_complete_path = LEADER_DIR / BATCH_AGGREGATION_COMPLETE_FILE
        self.monitor_path = LEADER_DIR / MONITOR_FILE
        for node in list_worker_nodes():
            self.training_complete_paths.append(self.get_path(node, TRAINING_COMPLETE_FILE))
            self.gradient_paths.append(self.get_path(node, GRADIENT_FILE))

    @staticmethod
    def get_path(worker_node: str, file: str) -> Path:
        return LEADER_DIR / worker_node / file

    def are_trainings_complete(self) -> bool:
        return all(os.path.exists(path) for path in self.training_complete_paths)

    async def wait_gradients(self):
        while not all(os.path.exists(path) for path in self.gradient_paths):
            await asyncio.sleep(WAITING_PERIOD)

    def delete_gradients(self):
        for path in self.gradient_paths:
            if os.path.exists(path):
                os.remove(path)

    def signal_batch_aggregation_complete(self):
        with open(self.batch_aggregation_complete_path, "wb"):
            pass

    def aggregate_gradients(self, network: nn.Module):
        gradients = [torch.load(path) for path in self.gradient_paths]
        avg_grads = gradients[0]
        num = len(gradients)

        for grad in gradients[1:]:
            for name, _ in network.named_parameters():
                avg_grads[name] = torch.add(avg_grads[name], grad[name])

        for name, param in network.named_parameters():
            param.grad = avg_grads[name] / num

    async def monitor(self, task: asyncio.Task):
        print("Leader monitor started.")
        while not task.done():
            with open(self.monitor_path, "wb"):
                pass
            print("Leader monitor: Leader is running.")
            await asyncio.sleep(MONITOR_PERIOD)

        print("Leader monitor finished.")

    async def aggregate_network(self):
        print("Leader started.")
        while True:
            await self.wait_gradients()

            network = load_network(AGGREGATED_STATE_DICT_PATH)
            optimizer = load_optimizer(network)

            self.aggregate_gradients(network)
            optimizer.step()
            optimizer.zero_grad()

            network_state_dict = network.state_dict()
            torch.save(network_state_dict, AGGREGATED_STATE_DICT_PATH)

            self.delete_gradients()
            self.signal_batch_aggregation_complete()

            if self.are_trainings_complete():
                print("Leader finished.")
                return

    async def run(self):
        aggregate_network_task = asyncio.create_task(self.aggregate_network())
        monitor_task = asyncio.create_task(self.monitor(aggregate_network_task))
        await asyncio.gather(aggregate_network_task, monitor_task)
