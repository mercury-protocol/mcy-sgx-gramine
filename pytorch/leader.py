import asyncio
import os
import torch
from torch import nn
from typing import Any

from pytorch.constants import WORKER_DIR, AGGREGATED_STATE_DICT_PATH, WAITING_PERIOD
from pytorch.utils import load_network, load_optimizer, get_file_path, DirEnum, FileEnum, safe_delete_file


def are_trainings_complete() -> bool:
    return all(
        os.path.exists(
            get_file_path(DirEnum.WORKER, node, FileEnum.TRAINING_COMPLETE)
        )
        for node
        in os.listdir(WORKER_DIR)
    )


def are_batch_trainings_complete() -> bool:
    return all(
        os.path.exists(
            get_file_path(DirEnum.WORKER, node, FileEnum.BATCH_TRAINING_COMPLETE)
        )
        for node
        in os.listdir(WORKER_DIR)
    )


def signal_batch_aggregations_complete():
    for node in os.listdir(WORKER_DIR):
        safe_delete_file(
            get_file_path(DirEnum.WORKER, node, FileEnum.BATCH_TRAINING_COMPLETE)
        )
        with open(
                get_file_path(DirEnum.WORKER, node, FileEnum.BATCH_AGGREGATION_COMPLETE),
                "wb"
        ):
            pass


def load_gradients() -> Any:
    gradients = list()
    for node in os.listdir(WORKER_DIR):
        path = get_file_path(DirEnum.WORKER, node, FileEnum.GRADIENT)
        gradients.append(torch.load(path))
    return gradients


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
        while not are_batch_trainings_complete():
            await asyncio.sleep(WAITING_PERIOD)

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
