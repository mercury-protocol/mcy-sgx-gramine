import asyncio
import os
import shutil

from pathlib import Path

from tests.constants import (
    TEMP_DIR,
    WORKER_FINISHED_FILE,
    USER_SCRIPT_FILE,
    STATE_DICT_READY_FILE,
    STATE_DICT_FILE,
    GRADIENT_READY_FILE,
    GRADIENT_FILE,
    WAITING_PERIOD,
)


def leader_dir() -> Path:
    return TEMP_DIR / "leader"


def leader_get_path(worker_node: str, file: str) -> Path:
    if "." in file:
        name, extension = file.split(".")
        file = f"{name}_{worker_node}.{extension}"
    else:
        file = f"{file}_{worker_node}"
    return leader_dir() / file


def worker_dir(node: str) -> Path:
    return TEMP_DIR / f"worker{node}"


def has_worker_finished(node: str) -> bool:
    return os.path.exists(worker_dir(node) / WORKER_FINISHED_FILE)


def list_worker_nodes(worker_count: int) -> list[str]:
    return [str(i + 1) for i in range(worker_count)]


class WatchLeader:
    def __init__(self, example_dir: Path, worker_count: int = 1):
        self.example_dir = example_dir
        self.worker_count = worker_count
        self.worker_nodes = list_worker_nodes(worker_count)
        self.state_dict_ready_path = leader_dir() / STATE_DICT_READY_FILE
        self.state_dict_path = leader_dir() / STATE_DICT_FILE

    def send_user_script_to_leader(self):
        shutil.copy(
            self.example_dir / USER_SCRIPT_FILE,
            leader_dir() / USER_SCRIPT_FILE
        )

    async def wait_state_dict(self):
        while not os.path.exists(self.state_dict_ready_path):
            await asyncio.sleep(WAITING_PERIOD)
        if not os.path.exists(self.state_dict_path):
            raise Exception(f"{self.state_dict_path} does not exist!")
        os.remove(self.state_dict_ready_path)

    def send_state_dict_to_worker(self, node: str):
        shutil.copy(
            self.state_dict_path,
            worker_dir(node) / STATE_DICT_FILE
        )
        with open(worker_dir(node) / STATE_DICT_READY_FILE, "wb"):
            pass

    async def run(self):
        if self.worker_count == 1:
            # if there's only one worker, leader is not needed
            return

        print("Watch leader started.")
        self.send_user_script_to_leader()
        while True:
            await self.wait_state_dict()

            for node in self.worker_nodes:
                self.send_state_dict_to_worker(node)

            if all(has_worker_finished(node) for node in self.worker_nodes):
                print("Watch leader finished.")
                return


class WatchWorker:
    def __init__(self, example_dir: Path, node: str, worker_count: int = 1):
        self.example_dir = example_dir
        self.node = node
        self.worker_count = worker_count
        self.worker_dir = worker_dir(self.node)

    def send_user_script_to_worker(self):
        shutil.copy(
            self.example_dir / USER_SCRIPT_FILE,
            self.worker_dir / USER_SCRIPT_FILE
        )

    def send_data_to_worker(self):
        data_dir = "data" if self.worker_count <= 1 else f"split_data/{self.node}"
        shutil.copytree(
            self.example_dir / data_dir,
            self.worker_dir / "data",
            dirs_exist_ok=True
        )

    async def wait_gradient(self):
        while not os.path.exists(self.worker_dir / GRADIENT_READY_FILE):
            await asyncio.sleep(WAITING_PERIOD)
        if not os.path.exists(self.worker_dir / GRADIENT_FILE):
            raise Exception(f"{self.worker_dir / GRADIENT_FILE} does not exist!")

    def send_worker_finished_to_leader(self):
        shutil.copy(
            self.worker_dir / WORKER_FINISHED_FILE,
            leader_get_path(self.node, WORKER_FINISHED_FILE)
        )

    def send_gradient_to_leader(self):
        shutil.copy(
            self.worker_dir / GRADIENT_FILE,
            leader_get_path(self.node, GRADIENT_FILE)
        )
        shutil.copy(
            self.worker_dir / GRADIENT_READY_FILE,
            leader_get_path(self.node, GRADIENT_READY_FILE)
        )
        os.remove(self.worker_dir / GRADIENT_FILE)
        os.remove(self.worker_dir / GRADIENT_READY_FILE)

    async def run(self):
        print(f"Watch worker {self.node} started")
        self.send_user_script_to_worker()
        self.send_data_to_worker()

        if self.worker_count == 1:
            # if there's only one worker, leader is not needed
            print(f"Watch worker {self.node} started")
            return

        while True:
            await self.wait_gradient()

            if has_worker_finished(self.node):
                self.send_worker_finished_to_leader()

            self.send_gradient_to_leader()

            if has_worker_finished(self.node):
                print(f"Watch worker {self.node} finished.")
                return


async def simulate_p2p_network_coroutine(example_dir: Path, worker_count: int = 1):
    watch_leader_task = asyncio.create_task(WatchLeader(example_dir, worker_count).run())
    watch_worker_tasks = [
        asyncio.create_task(WatchWorker(example_dir, node, worker_count).run())
        for node in list_worker_nodes(worker_count)
    ]
    await asyncio.gather(watch_leader_task, *watch_worker_tasks)


def simulate_p2p_network(example_dir: Path, worker_count: int = 1):
    asyncio.run(simulate_p2p_network_coroutine(example_dir, worker_count))
