import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader

from mcy_dist_ai.required_utils import DataSetFactory, DataLoaderFactory, NetworkFactory, OptimizerFactory


# ------------------- config ----------------
N_EPOCHS = 1
BATCH_SIZE_TRAIN = 500
LEARNING_RATE = 1e-3
MOMENTUM = 0.5

RANDOM_SEED = 1

torch.backends.cudnn.enabled = False
torch.manual_seed(RANDOM_SEED)

loss_fn = nn.CrossEntropyLoss() 


class PartitionedDataSetFactory(DataSetFactory):
    def create(self, data_path):
        partitioned_dataset = torch.load(data_path)
        return partitioned_dataset


# ------------------- build the network ----------------
class ImageClassifier(nn.Module): 
    def __init__(self):
        super(ImageClassifier, self).__init__()
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


# ------------------- create required objects ----------------
network_factory = NetworkFactory(ImageClassifier)
optimizer_factory = OptimizerFactory(Adam, lr=LEARNING_RATE)

data_set_factory = PartitionedDataSetFactory(None)

data_loader_factory = DataLoaderFactory(
    data_set_factory,
    DataLoader,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True)


# ------------------- train the model ----------------
def train_batch(data, target, network, optimizer):
    yhat = network(data)
    loss = loss_fn(yhat, target) 
    optimizer.zero_grad()
    loss.backward() 
    return loss  # for local logging purposes only
