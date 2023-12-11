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
from mcy_dist_ai import parse_node_num, export_gradients, wait_for_gradient_updates, complete_training
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
    mnist_dataset = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())

    train_size = int(0.8 * len(mnist_dataset))
    test_size = len(mnist_dataset) - train_size
    train_dataset, test_dataset = random_split(mnist_dataset, [train_size, test_size])

    # TODO: Split the dataset properly in the leader node and send chunk to the worker
    total_samples = len(train_dataset)
    samples_to_keep = total_samples // 2

    train = torch.utils.data.Subset(train_dataset, range(samples_to_keep))
    
    if node_num % 2 == 0:
        train = torch.utils.data.Subset(train_dataset, range(samples_to_keep, total_samples))

    dataset = DataLoader(train, 500)

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


"""     img = Image.open('img_3.jpg') 
    img_tensor = ToTensor()(img).unsqueeze(0)

    print(torch.argmax(clf(img_tensor))) """


"""             if current_batch % 100 == 0:
                gradients = [param.grad.clone().detach() if param.grad is not None else None for param in clf.parameters()]
                print("GRADIENTS: ")
                print(gradients)
                print("\n") """