import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# ------------------- config ----------------
N_EPOCHS = 1


# ------------------- create required functions ----------------
def create_model() -> nn.Module:
    pass


def create_optimizer(model: nn.Module) -> Optimizer:
    pass


def create_data_loader(path) -> DataLoader:
    pass


def create_extra_training_args(data_loader: DataLoader, optimizer: Optimizer):
    pass


def train_batch(batch, model, optimizer, *args):
    pass
