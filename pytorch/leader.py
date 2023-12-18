import os
import torch
from torch import nn

from pytorch.constants import SPLIT_NETWORK_PATH, AGGREGATED_NETWORK_PATH
from pytorch.utils import load_network, load_optimizer


def leader():
    gradient_update_paths = [f"{SPLIT_NETWORK_PATH}/{path}/network.pth" for path in os.listdir(SPLIT_NETWORK_PATH)]
    gradient_updates = [torch.load(path) for path in gradient_update_paths]

    network = load_network()
    optimizer = load_optimizer(network)
    avg_aggr_gradients(network=network, gradient_updates=gradient_updates)

    optimizer.step()
    optimizer.zero_grad()

    torch.save(network.state_dict(), AGGREGATED_NETWORK_PATH)


def avg_aggr_gradients(network: nn.Module, gradient_updates):
    aggregated_gradients = gradient_updates[0]
    for gradient in gradient_updates[1:]:
        for name, param in network.named_parameters():
            aggregated_gradients[name] = torch.add(aggregated_gradients[name], gradient[name])

    for name, param in network.named_parameters():
        aggregated_gradients[name] /= len(gradient_updates)
        param.grad = aggregated_gradients[name]


if __name__ == "__main__":
    leader()
