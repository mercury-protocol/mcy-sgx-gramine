import os
import torch

from user_script import train_batch, data_loader_factory, N_EPOCHS

from pytorch.constants import (
    SPLIT_DATA_PATH, SPLIT_NETWORK_PATH, AGGREGATED_STATE_DICT_PATH,
    STATE_DICT_PATH
)
from pytorch.helpers.eval import make_predictions
from pytorch.helpers.test_network import test, test_data_loader
from pytorch.utils import load_network, load_optimizer, save_gradients
from pytorch.leader import leader


LOG_INTERVAL = 10
train_losses = []
train_counter = []


def train_network():
    for dirname in os.listdir(SPLIT_DATA_PATH):
        data_path = f"{SPLIT_DATA_PATH}/{dirname}/"
        state_dict_path = f"{SPLIT_NETWORK_PATH}/{dirname}/state_dict.pth"
        optimizer_path = f"{SPLIT_NETWORK_PATH}/{dirname}/optimizer.pth"
        gradient_path = f"{SPLIT_NETWORK_PATH}/{dirname}/gradient.pth"

        data_loader = data_loader_factory.create(data_path)
        network = load_network(path=state_dict_path)
        optimizer = load_optimizer(network, path=optimizer_path)

        for epoch in range(N_EPOCHS):
            for batch_idx, (data, target) in enumerate(data_loader):
                loss = train_batch(data, target, network, optimizer)
                torch.save(network.state_dict(), state_dict_path)
                save_gradients(network, gradient_path)

                if batch_idx % LOG_INTERVAL == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(data_loader.dataset),
                        100. * batch_idx / len(data_loader), loss.item()))


def evaluate_network(state_dict_path=AGGREGATED_STATE_DICT_PATH):
    network = load_network(state_dict_path)
    test(network)
    make_predictions(network, test_data_loader)


if __name__ == "__main__":
    train_network()
    leader()
    evaluate_network()
