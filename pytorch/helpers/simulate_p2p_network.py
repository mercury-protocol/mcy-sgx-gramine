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
    WAITING_PERIOD,
    SPLIT_DATA_PATH,
    DATA_DIR
)
from pytorch.utils import list_worker_nodes


def is_training_complete(node: str) -> bool:
    return os.path.exists(WORKER_DIR / node / TRAINING_COMPLETE_FILE)


class LeaderPeer:
    def __init__(self):
        for node in list_worker_nodes():
            os.makedirs(LEADER_DIR / node, exist_ok=True)

    @staticmethod
    async def wait_network_aggregation():
        while not os.path.exists(LEADER_DIR / BATCH_AGGREGATION_COMPLETE_FILE):
            await asyncio.sleep(WAITING_PERIOD)
        os.remove(LEADER_DIR / BATCH_AGGREGATION_COMPLETE_FILE)

    @staticmethod
    def send_state_dict_to_worker(node: str):
        shutil.copy(
            AGGREGATED_STATE_DICT_PATH,
            WORKER_DIR / node / STATE_DICT_FILE
        )

    async def run(self):
        print("Watch leader started.")
        while True:
            await self.wait_network_aggregation()

            for node in list_worker_nodes():
                self.send_state_dict_to_worker(node)

            if all(is_training_complete(node) for node in list_worker_nodes()):
                print("Watch leader finished.")
                return


class WorkerPeer:
    def __init__(self, node: str):
        self.node = node

    def send_data_to_worker(self):
        shutil.copytree(
            SPLIT_DATA_PATH / self.node,
            WORKER_DIR / self.node / DATA_DIR,
            dirs_exist_ok=True
        )

    async def wait_batch_training(self):
        while not os.path.exists(WORKER_DIR / self.node / GRADIENT_FILE):
            await asyncio.sleep(WAITING_PERIOD)

    def signal_training_complete_to_leader(self):
        shutil.copy(
            WORKER_DIR / self.node / TRAINING_COMPLETE_FILE,
            LEADER_DIR / self.node / TRAINING_COMPLETE_FILE
        )

    def send_gradient_to_leader(self):
        shutil.move(
            WORKER_DIR / self.node / GRADIENT_FILE,
            LEADER_DIR / self.node / GRADIENT_FILE
        )

    async def run(self):
        print(f"Watch worker {self.node} started")
        self.send_data_to_worker()

        while True:
            await self.wait_batch_training()

            if is_training_complete(self.node):
                self.signal_training_complete_to_leader()

            self.send_gradient_to_leader()

            if is_training_complete(self.node):
                print(f"Watch worker {self.node} finished.")
                return


async def simulate_p2p_network():
    watch_leader_task = asyncio.create_task(LeaderPeer().run())
    watch_worker_tasks = [
        asyncio.create_task(WorkerPeer(node).run())
        for node in list_worker_nodes()
    ]
    await asyncio.gather(watch_leader_task, *watch_worker_tasks)
