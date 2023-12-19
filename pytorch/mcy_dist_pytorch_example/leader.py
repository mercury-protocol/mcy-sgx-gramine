
import os
import time
from torch import nn
from torch.optim import Adam
from mcy_dist_ai import parse_worker_nodes_count, aggregate_gradients, aggregate_gradients_and_save_model


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)


if __name__ == "__main__":
    worker_nodes_count = parse_worker_nodes_count()

    # Instance of the neural network, loss, optimizer 
    network = Network()
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
    aggregate_gradients_and_save_model(network, optimizer)

    
    