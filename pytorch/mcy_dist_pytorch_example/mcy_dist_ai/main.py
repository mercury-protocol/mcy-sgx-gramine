import torch
from torch import save, load
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import sys
import time
import argparse
from .data_partitioner import DataPartitioner

OUTPUT_MODEL_PATH = "/Users/lajosdeme/Desktop/"
OUTPUT_MODEL_FILE = "model_state.pt"


def aggregate_gradients(network, optimizer):
    listdir = os.listdir()

    gradient_update_files = [file for file in listdir if 'gradient_updates' in file]
    gradient_updates = [torch.load(file) for file in gradient_update_files]

    if len(gradient_updates) == 0:
        return
    
    avg_aggr_gradients(network=network, gradient_updates=gradient_updates)
            
    optimizer.step()
    optimizer.zero_grad()

    with open(OUTPUT_MODEL_FILE, 'wb') as f:
        save(network.state_dict(), f)

    for fname in gradient_update_files:
        os.remove(fname)

    model_path = f"{OUTPUT_MODEL_PATH}/trained_model1.pth"
    save(network.state_dict(), model_path)


def aggregate_gradients_and_save_model(network, optimizer):
    listdir = os.listdir()

    gradient_update_files = [file for file in listdir if 'gradient_last_updates' in file]
    gradient_updates = [torch.load(file) for file in gradient_update_files]

    if len(gradient_updates) == 0:
        return
    
    avg_aggr_gradients(network=network, gradient_updates=gradient_updates)
            
    optimizer.step()
    optimizer.zero_grad()

    model_path = f"{OUTPUT_MODEL_PATH}/trained_model.pth"
    save(network.state_dict(), model_path)

    for fname in gradient_update_files:
        os.remove(fname)


def avg_aggr_gradients(network, gradient_updates):
    aggregated_gradients = gradient_updates[0]
    for gradient in gradient_updates[1:]:
        for name, param in network.named_parameters():
            aggregated_gradients[name] = torch.add(aggregated_gradients[name], gradient[name])

    for name, param in network.named_parameters():
        aggregated_gradients[name] /= len(gradient_updates)
        param.grad = aggregated_gradients[name]


def parse_worker_args_test():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--worker_count", type=int, help="Worker nodes count")
    parser.add_argument("--node_num", type=int, help="Node number")
   
    args = parser.parse_args()

    if args.worker_count is None:
        print("Missing worker nodes count")
        sys.exit(1)

    if args.node_num is None:
        print("Missing node number")
        sys.exit(1)

    return args


def parse_worker_nodes_count():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_count", type=int, help="Worker nodes count")
    args = parser.parse_args()
    if args.worker_count is None:
        print("Missing worker nodes count")
        sys.exit(1)
    return args.worker_count


def parse_node_num():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_num", type=int, help="Node number")
    args = parser.parse_args()
    if args.node_num is None:
        print("Missing node number")
        sys.exit(1)
    return args.node_num


def export_gradients(model, node_num):
    gradients = {name: p.grad.data for name, p in model.named_parameters()}
    fname = f"gradient_updates_{node_num}.pt"
    with open(fname, 'wb') as f:
        save(gradients, f)
    return fname


def wait_for_gradient_updates(model):
    while not os.path.exists(OUTPUT_MODEL_FILE):
        time.sleep(1)

    with open(OUTPUT_MODEL_FILE, 'rb') as f:
        model.load_state_dict(load(f))
        os.remove(OUTPUT_MODEL_FILE)


def complete_training(network, node_num):
    fname = f"gradient_last_updates_{node_num}.pt"
    save(network.state_dict(), f"{OUTPUT_MODEL_PATH}/trained_{node_num}.pt")

    with open(fname, 'wb') as f:
        save(network.state_dict(), f)

    with open(f"training_complete_{node_num}", 'w'):
        pass


# TODO
def download_dataset():
    dataset = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
    return dataset


# Now dataset is split to equal sizes
# TODO: Calculate data chunk sizes according to node compute power and pass that in here
def partition_dataset(dataset, worker_nodes_count):
    partition_sizes = [1.0 / worker_nodes_count for _ in range(worker_nodes_count)]
    partition = DataPartitioner(dataset, partition_sizes)
    return partition


def get_data_partition_for_worker(partition, node_num):
    return partition.use(node_num-1)


def export_data_partitions(partitions, worker_nodes_count):
    for i in range(worker_nodes_count):
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        fname = f"partition_{i+1}.pth"
        save_path = os.path.join(parent_dir, fname)

        partition = partitions.use(i)

        save(partition, save_path)


def load_data(node_num):
    fname = f"partition_{node_num}.pth"
    return load(fname)
