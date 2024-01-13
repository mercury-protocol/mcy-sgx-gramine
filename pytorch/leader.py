import asyncio
import os
import torch
from torch import nn

from pytorch.constants import WORKER_DIR, AGGREGATED_STATE_DICT_PATH, WAITING_PERIOD
from pytorch.utils import load_network, load_optimizer, get_file_path, DirEnum, FileEnum, safe_delete_file


async def aggregate_network():
    print(f"Leader started.")
    gradient_paths = list()
    batch_aggregation_complete_paths = list()
    batch_training_complete_paths = list()
    training_complete_paths = list()
    for node in os.listdir(WORKER_DIR):
        gradient_paths.append(get_file_path(
            DirEnum.WORKER, node, FileEnum.GRADIENT))
        batch_aggregation_complete_paths.append(get_file_path(
            DirEnum.WORKER, node, FileEnum.BATCH_AGGREGATION_COMPLETE))
        batch_training_complete_paths.append(get_file_path(
            DirEnum.WORKER, node, FileEnum.BATCH_TRAINING_COMPLETE))
        training_complete_paths.append(get_file_path(
            DirEnum.WORKER, node, FileEnum.TRAINING_COMPLETE))

    while True:
        # wait for all workers to finish their training cycles
        while not all(os.path.exists(path) for path in batch_training_complete_paths):
            await asyncio.sleep(WAITING_PERIOD)

        gradients = [torch.load(path) for path in gradient_paths]

        network = load_network(AGGREGATED_STATE_DICT_PATH)
        optimizer = load_optimizer(network)

        aggregate_gradients(network, gradients)

        optimizer.step()
        optimizer.zero_grad()

        torch.save(network.state_dict(), AGGREGATED_STATE_DICT_PATH)

        for pth in batch_training_complete_paths:
            safe_delete_file(pth)
        for pth in batch_aggregation_complete_paths:
            with open(pth, "wb"):
                pass

        if all(os.path.exists(pth) for pth in training_complete_paths):
            print("All aggregations have been finished, leader stops")
            return


def aggregate_gradients(network: nn.Module, gradients):
    avg_grads = gradients[0]
    num = len(gradients)
    for grad in gradients[1:]:
        for name, _ in network.named_parameters():
            avg_grads[name] = torch.add(avg_grads[name], grad[name])

    for name, param in network.named_parameters():
        param.grad = avg_grads[name] / num


if __name__ == "__main__":
    asyncio.run(aggregate_network())
