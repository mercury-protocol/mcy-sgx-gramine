import asyncio
import os
import torch
from torch import nn

from pytorch.constants import (
    AGGREGATED_STATE_DICT_PATH,
    TRAINING_COMPLETE_FILE,
    STATE_DICT_FILE,
    GRADIENT_FILE,
    WAITING_PERIOD
)
from pytorch.utils import load_network, load_optimizer, list_worker_nodes, get_file_path


class Leader:
    def __init__(self):
        self.worker_training_complete_paths = list()
        self.worker_aggregated_state_dict_paths = list()
        self.worker_gradient_paths = list()
        for node in list_worker_nodes():
            self.worker_training_complete_paths.append(
                get_file_path(node, TRAINING_COMPLETE_FILE)
            )
            self.worker_aggregated_state_dict_paths.append(
                get_file_path(node, STATE_DICT_FILE)
            )
            self.worker_gradient_paths.append(
                get_file_path(node, GRADIENT_FILE)
            )

    def are_worker_trainings_complete(self) -> bool:
        return all(os.path.exists(path) for path in self.worker_training_complete_paths)

    async def wait_worker_gradients(self):
        while not all(os.path.exists(path) for path in self.worker_gradient_paths):
            await asyncio.sleep(WAITING_PERIOD)

    def delete_worker_gradients(self):
        for path in self.worker_gradient_paths:
            if os.path.exists(path):
                os.remove(path)

    def send_aggregated_network_to_workers(self, network_state_dict):
        for path in self.worker_aggregated_state_dict_paths:
            torch.save(network_state_dict, path)

    def aggregate_gradients(self, network: nn.Module):
        gradients = [torch.load(path) for path in self.worker_gradient_paths]
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
            await self.wait_worker_gradients()

            network = load_network(AGGREGATED_STATE_DICT_PATH)
            optimizer = load_optimizer(network)

            self.aggregate_gradients(network)
            optimizer.step()
            optimizer.zero_grad()

            network_state_dict = network.state_dict()
            torch.save(network_state_dict, AGGREGATED_STATE_DICT_PATH)

            self.delete_worker_gradients()
            self.send_aggregated_network_to_workers(network_state_dict)

            if self.are_worker_trainings_complete():
                print("All aggregations have been finished, leader stops")
                return
