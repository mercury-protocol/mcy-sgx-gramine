import asyncio
import os
import torch
from torch import nn
from typing import Any

from pytorch.constants import AGGREGATED_STATE_DICT_PATH, WAITING_PERIOD
from pytorch.utils import load_network, load_optimizer, list_worker_nodes, get_file_path, FileEnum, safe_delete_file


def are_trainings_complete() -> bool:
    return all(
        os.path.exists(get_file_path(node, FileEnum.TRAINING_COMPLETE))
        for node
        in list_worker_nodes()
    )


async def wait_batch_trainings_complete():
    while not all(
        os.path.exists(get_file_path(node, FileEnum.BATCH_TRAINING_COMPLETE))
        for node
        in list_worker_nodes()
    ):
        await asyncio.sleep(WAITING_PERIOD)


def signal_batch_aggregations_complete():
    for node in list_worker_nodes():
        safe_delete_file(get_file_path(node, FileEnum.BATCH_TRAINING_COMPLETE))
        with open(get_file_path(node, FileEnum.BATCH_AGGREGATION_COMPLETE), "wb"):
            pass


def load_gradients() -> Any:
    return [
        torch.load(get_file_path(node, FileEnum.GRADIENT))
        for node
        in list_worker_nodes()
    ]


def aggregate_gradients(network: nn.Module, gradients):
    avg_grads = gradients[0]
    num = len(gradients)
    for grad in gradients[1:]:
        for name, _ in network.named_parameters():
            avg_grads[name] = torch.add(avg_grads[name], grad[name])

    for name, param in network.named_parameters():
        param.grad = avg_grads[name] / num


async def aggregate_network():
    print(f"Leader started.")
    while True:
        await wait_batch_trainings_complete()

        gradients = load_gradients()
        network = load_network(AGGREGATED_STATE_DICT_PATH)
        optimizer = load_optimizer(network)

        aggregate_gradients(network, gradients)
        optimizer.step()
        optimizer.zero_grad()

        torch.save(network.state_dict(), AGGREGATED_STATE_DICT_PATH)
        signal_batch_aggregations_complete()

        if are_trainings_complete():
            print("All aggregations have been finished, leader stops")
            return


if __name__ == "__main__":
    asyncio.run(aggregate_network())
