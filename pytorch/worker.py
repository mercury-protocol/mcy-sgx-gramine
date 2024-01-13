import asyncio
import os
import torch

from user_script import train_batch, data_loader_factory, N_EPOCHS

from pytorch.constants import SPLIT_DATA_PATH, SPLIT_NETWORK_PATH, AGGREGATED_NETWORK_PATH, WAITING_PERIOD
from pytorch.utils import load_network, load_optimizer, save_gradients


LOG_INTERVAL = 10
train_losses = []
train_counter = []


async def train_network(dirname):
    print(f"Worker {dirname} started its training.")
    data_path = f"{SPLIT_DATA_PATH}/{dirname}/"
    state_dict_path = f"{SPLIT_NETWORK_PATH}/{dirname}/state_dict.pth"
    aggregated_state_dict_path = f"{AGGREGATED_NETWORK_PATH}/{dirname}/state_dict.pth"
    optimizer_path = f"{SPLIT_NETWORK_PATH}/{dirname}/optimizer.pth"
    gradient_path = f"{SPLIT_NETWORK_PATH}/{dirname}/gradient.pth"

    data_loader = data_loader_factory.create(data_path)
    network = load_network(path=aggregated_state_dict_path)
    optimizer = load_optimizer(network, path=optimizer_path)

    for epoch in range(N_EPOCHS):
        for batch_idx, (data, target) in enumerate(data_loader):
            loss = train_batch(data, target, network, optimizer)
            torch.save(network.state_dict(), state_dict_path)
            save_gradients(network, gradient_path)

            # wait other workers to finish and leader to aggregate
            while not os.path.exists(aggregated_state_dict_path):
                await asyncio.sleep(WAITING_PERIOD)

            network.load_state_dict(torch.load(aggregated_state_dict_path))
            os.remove(aggregated_state_dict_path)

            if batch_idx % LOG_INTERVAL == 0:
                print(f"Worker: {dirname} Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.6f}")

    # create a file to signal that training is complete
    with open(f"{SPLIT_NETWORK_PATH}/{dirname}/training_completed", "wb"):
        pass

    print(f"Worker {dirname} finished its training.")
