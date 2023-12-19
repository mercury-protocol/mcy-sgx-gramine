import os
import torch
from torch import nn

from pytorch.constants import SPLIT_NETWORK_PATH, AGGREGATED_STATE_DICT_PATH
from pytorch.utils import load_network, load_optimizer


def leader():
    gradient_paths = [f"{SPLIT_NETWORK_PATH}/{path}/gradient.pth" for path in os.listdir(SPLIT_NETWORK_PATH)]
    gradients = [torch.load(path) for path in gradient_paths]

    network = load_network(AGGREGATED_STATE_DICT_PATH)
    optimizer = load_optimizer(network)

    aggregate_gradients(network, gradients)

    optimizer.step()
    optimizer.zero_grad()

    torch.save(network.state_dict(), AGGREGATED_STATE_DICT_PATH)


def aggregate_gradients(network: nn.Module, gradients):
    avg_grads = gradients[0]
    num = len(gradients)
    for grad in gradients[1:]:
        for name, _ in network.named_parameters():
            avg_grads[name] = torch.add(avg_grads[name], grad[name])

    for name, param in network.named_parameters():
        param.grad = avg_grads[name] / num


def _leader():
    paths = [f"{SPLIT_NETWORK_PATH}/{path}/state_dict.pth" for path in os.listdir(SPLIT_NETWORK_PATH)]
    state_dicts = [torch.load(path) for path in paths]
    num = len(state_dicts)

    aggregate = dict()
    for key in state_dicts[0]:
        aggregate[key] = sum(sd[key] for sd in state_dicts) / num

    torch.save(aggregate, AGGREGATED_STATE_DICT_PATH)


if __name__ == "__main__":
    leader()
