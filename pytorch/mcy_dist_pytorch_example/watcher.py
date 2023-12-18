from mcy_dist_ai import parse_worker_nodes_count, download_dataset, partition_dataset, export_data_partitions

if __name__ == "__main__":
    world_size = parse_worker_nodes_count()
    dataset = download_dataset()
    partitioned_dataset = partition_dataset(dataset=dataset, worker_nodes_count=world_size)
    export_data_partitions(partitions=partitioned_dataset, worker_nodes_count=world_size)