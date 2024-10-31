import os
import torch
from torch.utils.data import Dataset, DataLoader

from mcy_dist_ai.import_user_files import import_user_script


class LabeledTensorDataset(Dataset):
    def __init__(self, data_tensor, label_tensor):
        self.data = data_tensor
        self.labels = label_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx], "labels": self.labels[idx]}


def create_labeled_tensor_loader(data_path: str, user_script_path: str):
    user_script = import_user_script(user_script_path)

    data_tensor = torch.load(f"{data_path}/data_tensor.pt")
    label_tensor = torch.load(f"{data_path}/label_tensor.pt")
    tensor_dataset = LabeledTensorDataset(data_tensor, label_tensor)

    return DataLoader(
        tensor_dataset,
        batch_size=user_script.BATCH_SIZE,
        shuffle=True
    )


def split_labeled_data(split_into: int, data_path: str, output_dir_path: str, user_script_path: str):
    # TODO: handle large data memory-efficiently
    all_data = []
    all_labels = []

    user_script = import_user_script(user_script_path)
    original_data_loader = user_script.create_data_loader(data_path)

    for batch in original_data_loader:
        all_data.append(batch["input_ids"])
        all_labels.append(batch["labels"])

    all_data = torch.cat(all_data)
    all_labels = torch.cat(all_labels)

    partition_size = len(all_data) // split_into
    data_partitions = torch.split(all_data, partition_size)
    label_partitions = torch.split(all_labels, partition_size)

    for i, (data_part, label_part) in enumerate(zip(data_partitions, label_partitions), start=1):
        os.makedirs(f"{output_dir_path}/{i}", exist_ok=True)

        data_file = f"{output_dir_path}/{i}/data_tensor.pt"
        label_file = f"{output_dir_path}/{i}/label_tensor.pt"

        torch.save(data_part, data_file)
        torch.save(label_part, label_file)

