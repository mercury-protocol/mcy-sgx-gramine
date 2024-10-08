import os
import torch
from torch.utils.data import Dataset, DataLoader

from mcy_dist_ai.constants import DATA_PATH, PARTITIONED_TENSORS_PATH, WORKER_NODES_NUM
from mcy_dist_ai.utils import user_script


class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data = data_tensor
        self.targets = target_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def split_and_save_tensors():
    # TODO: handle large data memory-efficiently
    all_data = []
    all_targets = []

    original_data_loader = user_script.create_data_loader(str(DATA_PATH))
    for batch_data, batch_targets in original_data_loader:
        all_data.append(batch_data)
        all_targets.append(batch_targets)

    all_data = torch.cat(all_data)
    all_targets = torch.cat(all_targets)

    partition_size = len(all_data) // WORKER_NODES_NUM
    data_partitions = torch.split(all_data, partition_size)
    target_partitions = torch.split(all_targets, partition_size)

    os.makedirs(PARTITIONED_TENSORS_PATH, exist_ok=True)

    for i, (data_part, target_part) in enumerate(zip(data_partitions, target_partitions)):
        data_file = PARTITIONED_TENSORS_PATH / f"data_partition_{i + 1}.pt"
        target_file = PARTITIONED_TENSORS_PATH / f"target_partition_{i + 1}.pt"

        torch.save(data_part, data_file)
        torch.save(target_part, target_file)


def create_tensor_loader(partition_index):
    data_tensor = torch.load(PARTITIONED_TENSORS_PATH / f"data_partition_{partition_index}.pt")
    target_tensor = torch.load(PARTITIONED_TENSORS_PATH / f"target_partition_{partition_index}.pt")

    tensor_dataset = TensorDataset(data_tensor, target_tensor)

    return DataLoader(
        tensor_dataset,
        batch_size=user_script.BATCH_SIZE,
        shuffle=True
    )
