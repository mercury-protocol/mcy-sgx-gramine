import os
import time
import torch 
import argparse
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from mcy_dist_ai import export_gradients, wait_for_gradient_updates, complete_training, partition_dataset, get_data_partition_for_worker, parse_node_num, load_data
from torch.utils.data import random_split

import shutil

# Image Classifier Neural Network
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

# Instance of the neural network, loss, optimizer 
clf = ImageClassifier()
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

# Training flow 
if __name__ == "__main__": 
    node_num = parse_node_num()
    # Get data 
    # mnist_dataset = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
    
    # TODO: Split the dataset in the watcher node and send chunk to the worker
    # partitioned_dataset = partition_dataset(mnist_dataset, world_size)
    # partition = get_data_partition_for_worker(partitioned_dataset, node_num)
    partitioned_dataset = load_data(node_num=node_num)

    dataset = DataLoader(partitioned_dataset, 500)

    for epoch in range(1):  # train for 10 epochs
        current_batch = 0
        for batch in dataset:             
            X, y = batch 
            #X, y = X.to('cuda'), y.to('cuda') 
            yhat = clf(X) 
            loss = loss_fn(yhat, y) 

            # Apply backprop 
            opt.zero_grad()
            loss.backward() 
            # opt.step()
            #if current_batch % 100 == 0:
            export_gradients(clf, node_num)
            wait_for_gradient_updates(clf, node_num)

            current_batch += 1

        print(f"Epoch: {epoch} loss is {loss.item()}")

    complete_training(clf, node_num)