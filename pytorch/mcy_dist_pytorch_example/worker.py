from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from mcy_dist_ai import export_gradients, wait_for_gradient_updates, complete_training, parse_node_num, load_data
from utils import ImageClassifierNetwork


network = ImageClassifierNetwork()
optimizer = Adam(network.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

# Training flow 
if __name__ == "__main__": 
    node_num = parse_node_num()

    # TODO: Split the dataset in the watcher node and send chunk to the worker
    partitioned_dataset = load_data(node_num=node_num)

    dataset = DataLoader(partitioned_dataset, 500)

    for epoch in range(1):  # train for 10 epochs
        current_batch = 0
        for batch in dataset:             
            X, y = batch
            yhat = network(X)
            loss = loss_fn(yhat, y) 

            # Apply backprop 
            optimizer.zero_grad()
            loss.backward()
            export_gradients(network, node_num)
            wait_for_gradient_updates(network)

            current_batch += 1

        print(f"Epoch: {epoch} loss is {loss.item()}")

    complete_training(network, node_num)
