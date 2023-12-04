import os
import torch

from constants import SPLIT_NETWORK_PATH, AGGREGATED_NETWORK_PATH


def leader():
    gradient_update_paths = [f"{SPLIT_NETWORK_PATH}/{path}/network.pth" for path in os.listdir(SPLIT_NETWORK_PATH)]

    state_dict_list = []
    for path in gradient_update_paths:
        with open(path, 'rb') as f:
            f.seek(0)
            state_dict = torch.load(f)
            state_dict_list.append(state_dict)

    for params in zip(*[state_dict.values() for state_dict in state_dict_list]):
        params[0].add_(*params[1:])

    torch.save(state_dict_list[0], AGGREGATED_NETWORK_PATH)


if __name__ == "__main__":
    leader()
