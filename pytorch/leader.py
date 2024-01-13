import asyncio
import os
import torch
from torch import nn

from pytorch.constants import (
    AGGREGATED_STATE_DICT_PATH,
    TRAINING_COMPLETE_FILE,
    BATCH_TRAINING_COMPLETE_FILE,
    BATCH_AGGREGATION_COMPLETE,
    GRADIENT_FILE,
    WAITING_PERIOD
)
from pytorch.utils import load_network, load_optimizer, list_worker_nodes, get_file_path, safe_delete_file


class Leader:
    def __init__(self):
        self.training_complete_paths = list()
        self.batch_training_complete_paths = list()
        self.batch_aggregation_complete_paths = list()
        self.gradient_paths = list()
        for node in list_worker_nodes():
            self.training_complete_paths.append(
                get_file_path(node, TRAINING_COMPLETE_FILE)
            )
            self.batch_training_complete_paths.append(
                get_file_path(node, BATCH_TRAINING_COMPLETE_FILE)
            )
            self.batch_aggregation_complete_paths.append(
                get_file_path(node, BATCH_AGGREGATION_COMPLETE)
            )
            self.gradient_paths.append(
                get_file_path(node, GRADIENT_FILE)
            )

    def are_trainings_complete(self) -> bool:
        return all(os.path.exists(path) for path in self.training_complete_paths)

    async def wait_batch_trainings_complete(self):
        while not all(os.path.exists(path) for path in self.batch_training_complete_paths):
            await asyncio.sleep(WAITING_PERIOD)

    def signal_batch_aggregations_complete(self):
        for path in self.batch_training_complete_paths:
            safe_delete_file(path)
        for path in self.batch_aggregation_complete_paths:
            with open(path, "wb"):
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

    async def aggregate_network(self):
        print(f"Leader started.")
        while True:
            await self.wait_batch_trainings_complete()

            network = load_network(AGGREGATED_STATE_DICT_PATH)
            optimizer = load_optimizer(network)

            self.aggregate_gradients(network)
            optimizer.step()
            optimizer.zero_grad()

            torch.save(network.state_dict(), AGGREGATED_STATE_DICT_PATH)
            self.signal_batch_aggregations_complete()

            if self.are_trainings_complete():
                print("All aggregations have been finished, leader stops")
                return
