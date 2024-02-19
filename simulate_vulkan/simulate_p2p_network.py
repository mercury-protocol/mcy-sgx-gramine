import asyncio
import os
import shutil
from docker.models.containers import Container

from pytorch.constants import (
    AGGREGATED_STATE_DICT_PATH,
    LEADER_DIR,
    STATE_DICT_READY_FILE,
    WORKER_DIR,
    USER_SCRIPT_FILE,
    STATE_DICT_FILE,
    GRADIENT_FILE,
    GRADIENT_READY_FILE,
    WORKER_FINISHED_FILE,
    WAITING_PERIOD
)

from simulate_vulkan.constants import LOCAL_SPLIT_DATA_PATH, LOCAL_USER_SCRIPT_PATH
from simulate_vulkan.docker_adapter import (
    copy_file_to_container, create_empty_file_in_container, delete_file_in_container, list_running_containers_from_image
)
from simulate_vulkan.utils import list_worker_nodes, leader_get_path


def has_worker_finished(node: str) -> bool:
    return os.path.exists(WORKER_DIR / node / WORKER_FINISHED_FILE)


class WatchLeader:
    def __init__(self, container: Container):
        self.container = container
        self.worker_containers = list_running_containers_from_image("worker:latest")

    def send_user_script_to_leader(self):
        copy_file_to_container(
            self.container,
            LOCAL_USER_SCRIPT_PATH,
            LEADER_DIR / USER_SCRIPT_FILE
        )

    async def wait_state_dict(self):
        while not os.path.exists(LEADER_DIR / STATE_DICT_READY_FILE):
            await asyncio.sleep(WAITING_PERIOD)
        if not os.path.exists(AGGREGATED_STATE_DICT_PATH):
            raise Exception(f"{AGGREGATED_STATE_DICT_PATH} does not exist!")
        delete_file_in_container(self.container, LEADER_DIR / STATE_DICT_READY_FILE)

    @staticmethod
    def send_state_dict_to_worker(worker_container):
        copy_file_to_container(
            worker_container,
            AGGREGATED_STATE_DICT_PATH,
            WORKER_DIR / STATE_DICT_FILE
        )
        create_empty_file_in_container(worker_container, WORKER_DIR / STATE_DICT_READY_FILE)

    async def run(self):
        print("Watch leader started.")
        self.send_user_script_to_leader()
        while True:
            await self.wait_state_dict()

            for container in self.worker_containers:
                self.send_state_dict_to_worker(container)

            if all(has_worker_finished(node) for node in list_worker_nodes()):
                print("Watch leader finished.")
                return


class WatchWorker:
    def __init__(self, container: Container, node: str):
        self.container = container
        self.node = node
        self.leader_container = list_running_containers_from_image("leader:latest")[0]

    def send_user_script_to_worker(self):
        copy_file_to_container(
            self.container,
            LOCAL_USER_SCRIPT_PATH,
            WORKER_DIR / USER_SCRIPT_FILE
        )

    def send_data_to_worker(self):
        shutil.copytree(
            LOCAL_SPLIT_DATA_PATH / self.node,
            WORKER_DIR / self.node / "data",
            dirs_exist_ok=True
        )

    async def wait_gradient(self):
        while not os.path.exists(WORKER_DIR / self.node / GRADIENT_READY_FILE):
            await asyncio.sleep(WAITING_PERIOD)
        if not os.path.exists(WORKER_DIR / self.node / GRADIENT_FILE):
            raise Exception(f"Gradient file in worker {self.node} does not exist!")

    def send_worker_finished_to_leader(self):
        copy_file_to_container(
            self.leader_container,
            WORKER_DIR / self.node / WORKER_FINISHED_FILE,
            leader_get_path(self.node, WORKER_FINISHED_FILE)
        )

    def send_gradient_to_leader(self):
        copy_file_to_container(
            self.leader_container,
            WORKER_DIR / self.node / GRADIENT_FILE,
            leader_get_path(self.node, GRADIENT_FILE)
        )
        copy_file_to_container(
            self.leader_container,
            WORKER_DIR / self.node / GRADIENT_READY_FILE,
            leader_get_path(self.node, GRADIENT_READY_FILE)
        )
        delete_file_in_container(self.container, WORKER_DIR / GRADIENT_FILE)
        delete_file_in_container(self.container, WORKER_DIR / GRADIENT_READY_FILE)

    async def run(self):
        print(f"Watch worker {self.node} started")
        self.send_user_script_to_worker()
        self.send_data_to_worker()

        while True:
            await self.wait_gradient()

            if has_worker_finished(self.node):
                self.send_worker_finished_to_leader()

            self.send_gradient_to_leader()

            if has_worker_finished(self.node):
                print(f"Watch worker {self.node} finished.")
                return


async def simulate_p2p_network(container_mapping: dict):
    watch_leader_task = asyncio.create_task(WatchLeader(container_mapping["leader"]).run())
    watch_worker_tasks = [
        asyncio.create_task(WatchWorker(container_mapping[f"worker_{node}"], node).run())
        for node in list_worker_nodes()
    ]
    await asyncio.gather(watch_leader_task, *watch_worker_tasks)
