import asyncio
import os
import torch
from torch import nn

from pytorch.constants import SPLIT_NETWORK_PATH, AGGREGATED_NETWORK_PATH, AGGREGATED_STATE_DICT_PATH, WAITING_PERIOD
from pytorch.utils import load_network, load_optimizer


async def aggregate_network():
    print(f"Leader started.")
    while True:
        gradient_paths = list()
        aggregation_completed_paths = list()
        for dirname in os.listdir(SPLIT_NETWORK_PATH):
            grad_pth = f"{SPLIT_NETWORK_PATH}/{dirname}/gradient.pth"
            aggr_compl_pth = f"{AGGREGATED_NETWORK_PATH}/{dirname}/aggregation_completed"
            gradient_paths.append(grad_pth)
            aggregation_completed_paths.append(aggr_compl_pth)

        # wait for all workers to finish their training cycles
        while not all([os.path.exists(path) for path in gradient_paths]):
            await asyncio.sleep(WAITING_PERIOD)

        gradients = [torch.load(path) for path in gradient_paths]

        network = load_network(AGGREGATED_STATE_DICT_PATH)
        optimizer = load_optimizer(network)

        aggregate_gradients(network, gradients)

        optimizer.step()
        optimizer.zero_grad()

        torch.save(network.state_dict(), AGGREGATED_STATE_DICT_PATH)
        for dirname in os.listdir(AGGREGATED_NETWORK_PATH):
            torch.save(network.state_dict(), f"{AGGREGATED_NETWORK_PATH}/{dirname}/state_dict.pth")

        for dirname in os.listdir(SPLIT_NETWORK_PATH):
            grad_pth = f"{SPLIT_NETWORK_PATH}/{dirname}/gradient.pth"
            tr_compl_pth = f"{SPLIT_NETWORK_PATH}/{dirname}/training_completed"
            aggr_compl_pth = f"{AGGREGATED_NETWORK_PATH}/{dirname}/aggregation_completed"
            os.remove(grad_pth)
            if os.path.exists(tr_compl_pth):
                with open(aggr_compl_pth, "wb"):
                    pass

        if all([os.path.exists(path) for path in aggregation_completed_paths]):
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
