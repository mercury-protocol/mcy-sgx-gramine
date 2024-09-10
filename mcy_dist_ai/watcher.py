import argparse
import sys
import torch

from mcy_dist_ai.data_partitioner import DataPartitioner
from mcy_dist_ai.constants import WATCHER_DATA_PATH
from mcy_dist_ai.logger import logger


def parse_worker_nodes_count():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_count", type=int, help="Worker nodes count")
    args = parser.parse_args()
    if args.worker_count is None:
        logger.error("Missing worker nodes count")
        sys.exit(1)
    return args.worker_count


def download_dataset():
    return torch.load(WATCHER_DATA_PATH)


# TODO: Calculate data chunk sizes according to node compute power and pass that in here
def partition_dataset(dataset, worker_nodes_count):
    partition_sizes = [1.0 / worker_nodes_count] * worker_nodes_count
    partition = DataPartitioner(dataset, partition_sizes)
    return partition


def export_data_partitions(partitions, worker_nodes_count):
    for n in range(worker_nodes_count):
        fname = f"partition_{n + 1}.pth"
        partition = partitions.use(n)
        torch.save(partition, fname)


if __name__ == "__main__":
    world_size = parse_worker_nodes_count()
    dataset = download_dataset()
    partitioned_dataset = partition_dataset(dataset=dataset, worker_nodes_count=world_size)
    export_data_partitions(partitions=partitioned_dataset, worker_nodes_count=world_size)
