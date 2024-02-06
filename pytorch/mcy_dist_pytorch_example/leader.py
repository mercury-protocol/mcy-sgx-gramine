import os
import time
from torch.optim import Adam
from mcy_dist_ai import parse_worker_nodes_count, aggregate_gradients
from utils import ImageClassifierNetwork


if __name__ == "__main__":
    worker_nodes_count = parse_worker_nodes_count()

    network = ImageClassifierNetwork()
    optimizer = Adam(network.parameters(), lr=1e-3)

    while not os.path.exists("training_complete"):
        listdir = os.listdir()
        gradient_update_files = [file for file in listdir if 'gradient_updates' in file]

        while len(gradient_update_files) != worker_nodes_count and not os.path.exists("training_complete"):
            listdir = os.listdir()
            gradient_update_files = [file for file in listdir if 'gradient_updates' in file]

            time.sleep(1)

        aggregate_gradients(network, optimizer)

    # aggregate last updates
    time.sleep(5)
    aggregate_gradients(network, optimizer, last_update=True)

    
    