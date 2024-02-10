import asyncio
import os
from time import time

from pytorch.constants import AGGREGATED_STATE_DICT_PATH, LEADER_DIR, WORKER_DIR, MONITOR_FILE, LEADER_ROLE, WORKER_ROLE
from pytorch.utils import load_network, list_worker_nodes

from remote.utils import MeasureTime
from simulate_vulkan.docker_adapter import create_container
from simulate_vulkan.helpers.eval import evaluate_network
from simulate_vulkan.simulate_p2p_network import simulate_p2p_network


def check_output_file_ages():
    now = time()
    aggregated_network_age = now - os.path.getmtime(AGGREGATED_STATE_DICT_PATH)
    leader_monitor_age = now - os.path.getmtime(LEADER_DIR / MONITOR_FILE)
    worker_monitor_ages = [
        now - os.path.getmtime(WORKER_DIR / node / MONITOR_FILE)
        for node in list_worker_nodes()
    ]
    print(f"aggregated network age: {aggregated_network_age}")
    print(f"leader monitor age: {leader_monitor_age}")
    print(f"worker monitor ages: {worker_monitor_ages}")


def main():
    create_container(LEADER_ROLE, worker_nodes_num=2)
    create_container(WORKER_ROLE, node=0)
    create_container(WORKER_ROLE, node=1)
    asyncio.run(simulate_p2p_network())


if __name__ == "__main__":
    with MeasureTime("train_network()"):
        main()

    check_output_file_ages()
    network = load_network(AGGREGATED_STATE_DICT_PATH)
    evaluate_network(network)
    # make_predictions(network, test_data_loader)
