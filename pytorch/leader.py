import os
import torch
from torch import nn

from pytorch.constants import SPLIT_NETWORK_PATH, AGGREGATED_NETWORK_PATH
from pytorch.utils import load_network, load_optimizer


def leader():
    gradient_paths = [f"{SPLIT_NETWORK_PATH}/{path}/network.pth" for path in os.listdir(SPLIT_NETWORK_PATH)]
    gradients = [torch.load(path) for path in gradient_paths]

    network = load_network()
    optimizer = load_optimizer(network)

    aggregated_gradients = aggregate_gradients(network, gradients)
    for name, param in network.named_parameters():
        param.grad = aggregated_gradients[name]

    optimizer.step()
    optimizer.zero_grad()

    torch.save(network.state_dict(), AGGREGATED_NETWORK_PATH)


def aggregate_gradients(network: nn.Module, gradients):
    aggregated_gradients = gradients[0]
    len_gradients = len(gradients)
    for gradient in gradients[1:]:
        for name, _ in network.named_parameters():
            aggregated_gradients[name] = torch.add(aggregated_gradients[name], gradient[name])
            aggregated_gradients[name] /= len_gradients

    return aggregated_gradients


if __name__ == "__main__":
    leader()
