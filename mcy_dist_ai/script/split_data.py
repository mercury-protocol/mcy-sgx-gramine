import os
import sys
import torch

from mcy_dist_ai.import_user_files import import_user_script


def split_data(split_into: int, data_path: str, output_dir_path: str, user_script_path: str):
    # TODO: handle large data memory-efficiently
    all_data = []
    all_targets = []

    user_script = import_user_script(user_script_path)
    original_data_loader = user_script.create_data_loader(data_path)
    for batch_data, batch_targets in original_data_loader:
        all_data.append(batch_data)
        all_targets.append(batch_targets)

    all_data = torch.cat(all_data)
    all_targets = torch.cat(all_targets)

    partition_size = len(all_data) // split_into
    data_partitions = torch.split(all_data, partition_size)
    target_partitions = torch.split(all_targets, partition_size)

    for i, (data_part, target_part) in enumerate(zip(data_partitions, target_partitions), start=1):
        os.makedirs(f"{output_dir_path}/{i}", exist_ok=True)

        data_file = f"{output_dir_path}/{i}/data_tensor.pt"
        target_file = f"{output_dir_path}/{i}/target_tensor.pt"

        torch.save(data_part, data_file)
        torch.save(target_part, target_file)


if __name__ == "__main__":
    split_data(
        int(sys.argv[1]),
        sys.argv[2],
        sys.argv[3],
        sys.argv[4]
    )
