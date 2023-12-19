import os

from user_script import train, data_loader_factory

from pytorch.constants import (
    SPLIT_DATA_PATH, SPLIT_NETWORK_PATH, AGGREGATED_STATE_DICT_PATH
)
from pytorch.helpers.eval import make_predictions
from pytorch.helpers.test_network import test, test_data_loader
from pytorch.utils import load_network, load_optimizer, save_gradients
from pytorch.leader import leader


def train_network():
    for dirname in os.listdir(SPLIT_DATA_PATH):
        data_path = f"{SPLIT_DATA_PATH}/{dirname}/"
        state_dict_path = f"{SPLIT_NETWORK_PATH}/{dirname}/state_dict.pth"
        optimizer_path = f"{SPLIT_NETWORK_PATH}/{dirname}/optimizer.pth"
        gradient_path = f"{SPLIT_NETWORK_PATH}/{dirname}/gradient.pth"

        data_loader = data_loader_factory.create(data_path)
        network = load_network(path=state_dict_path)
        optimizer = load_optimizer(network, path=optimizer_path)

        train(data_loader, network, optimizer)

        save_gradients(network, gradient_path)


def evaluate_network():
    network = load_network(AGGREGATED_STATE_DICT_PATH)
    test(network)
    make_predictions(network, test_data_loader)


if __name__ == "__main__":
    train_network()
    leader()
    evaluate_network()
