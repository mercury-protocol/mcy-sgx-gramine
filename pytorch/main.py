import asyncio
import os
from time import time

from pytorch.constants import AGGREGATED_STATE_DICT_PATH, LEADER_DIR, WORKER_DIR, MONITOR_FILE
from pytorch.helpers.eval import evaluate_network, make_predictions, test_data_loader
from pytorch.helpers.simulate_p2p_network import simulate_p2p_network
from pytorch.utils import load_network, list_worker_nodes
from pytorch.worker import Worker
from pytorch.leader import Leader
from remote.utils import MeasureTime


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


async def main():
    p2p_network_task = asyncio.create_task(simulate_p2p_network())
    leader_task = asyncio.create_task(Leader().run())
    worker_tasks = [
        asyncio.create_task(Worker(node).run())
        for node in list_worker_nodes()
    ]
    await asyncio.gather(p2p_network_task, leader_task, *worker_tasks)


if __name__ == "__main__":
    with MeasureTime("train_network()"):
        asyncio.run(main())

    check_output_file_ages()
    network = load_network(AGGREGATED_STATE_DICT_PATH)
    evaluate_network(network)
    # make_predictions(network, test_data_loader)
