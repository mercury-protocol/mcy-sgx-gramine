import asyncio
import os
import shutil

from pytorch.constants import (
    AGGREGATED_STATE_DICT_PATH,
    LEADER_DIR,
    BATCH_AGGREGATION_COMPLETE_FILE,
    WORKER_DIR,
    STATE_DICT_FILE,
    GRADIENT_FILE,
    TRAINING_COMPLETE_FILE,
    WAITING_PERIOD
)
from pytorch.utils import list_worker_nodes


def is_training_complete(worker_node: str) -> bool:
    return os.path.exists(WORKER_DIR / worker_node / TRAINING_COMPLETE_FILE)


async def wait_network_aggregation():
    while not os.path.exists(LEADER_DIR / BATCH_AGGREGATION_COMPLETE_FILE):
        await asyncio.sleep(WAITING_PERIOD)
    os.remove(LEADER_DIR / BATCH_AGGREGATION_COMPLETE_FILE)


async def wait_gradients(gradient_path):
    while not os.path.exists(gradient_path):
        await asyncio.sleep(WAITING_PERIOD)


async def watch_leader():
    print("Watch leader started.")
    while True:
        await wait_network_aggregation()

        for node in list_worker_nodes():
            shutil.copy(
                AGGREGATED_STATE_DICT_PATH,
                WORKER_DIR / node / STATE_DICT_FILE
            )

        if all(is_training_complete(node) for node in list_worker_nodes()):
            print("Watch leader finished.")
            return


async def watch_worker(node):
    print(f"Watch worker {node} started")
    while True:
        gradient_path = WORKER_DIR / node / GRADIENT_FILE
        await wait_gradients(gradient_path)

        os.makedirs(LEADER_DIR / node, exist_ok=True)

        if is_training_complete(node):
            shutil.copy(
                WORKER_DIR / node / TRAINING_COMPLETE_FILE,
                LEADER_DIR / node / TRAINING_COMPLETE_FILE
            )

        shutil.move(
            gradient_path,
            LEADER_DIR / node / GRADIENT_FILE
        )

        if is_training_complete(node):
            print(f"Watch worker {node} finished.")
            return


async def simulate_p2p_network():
    watch_leader_task = asyncio.create_task(watch_leader())
    watch_worker_tasks = [
        asyncio.create_task(watch_worker(node))
        for node in list_worker_nodes()
    ]
    await asyncio.gather(watch_leader_task, *watch_worker_tasks)
