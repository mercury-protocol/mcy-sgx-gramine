import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from mcy_dist_ai import parse_node_num, export_gradients, wait_for_gradient_updates, complete_training


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
    # TODO: Split the dataset properly in the leader node and send chunk to the worker
    total_samples = len(mnist_dataset)
    samples_to_keep = total_samples // 2

    train = torch.utils.data.Subset(mnist_dataset, range(samples_to_keep))
    
    if node_num % 2 == 0:
        train = torch.utils.data.Subset(mnist_dataset, range(samples_to_keep, total_samples))

    dataset = DataLoader(train, 32)

    for epoch in range(2):  # train for 10 epochs
        current_batch = 0
        for batch in dataset:             
            X, y = batch
            yhat = clf(X) 
            loss = loss_fn(yhat, y) 

            # Apply backprop 
            opt.zero_grad()
            loss.backward() 
            opt.step() 

            # Export gradient updates
            if current_batch % 900 == 0:
                export_gradients(clf, node_num)
                clf = wait_for_gradient_updates(clf, node_num)
            
            current_batch += 1

        print(f"Epoch: {epoch} loss is {loss.item()}")

    complete_training(clf, node_num)
