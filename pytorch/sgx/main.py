import pickle
import torch
from data import test_loader
from network import INITIAL_NETWORK_PATH, INITIAL_OPTIMIZER_PATH
from train import train

from pytorch.external_constants import MODEL_PATH, OPTIMIZER_PATH
from pytorch.normal.eval import make_predictions


N_EPOCHS = 2


def load_network():
    with open(INITIAL_NETWORK_PATH, "rb") as file:
        network = pickle.load(file)
    network.load_state_dict(torch.load(MODEL_PATH))
    return network


def load_optimizer():
    with open(INITIAL_OPTIMIZER_PATH, "rb") as file:
        optimizer = pickle.load(file)
    optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
    return optimizer


# ------------------- train the model ----------------
for _ in range(N_EPOCHS):
    _network = load_network()
    _optimizer = load_optimizer()

    train(_network, _optimizer)

    torch.save(_network.state_dict(), MODEL_PATH)
    torch.save(_optimizer.state_dict(), OPTIMIZER_PATH)

# ------------------- evaluate the better model ----------------
final_network = load_network()
make_predictions(final_network, test_loader)
