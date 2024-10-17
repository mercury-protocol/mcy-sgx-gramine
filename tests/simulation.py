import asyncio
import multiprocessing
import os
import shutil

from pathlib import Path
from unittest.mock import patch

from tests.constants import (
    TEMP_DIR,
    WORKER_FINISHED_FILE,
    USER_SCRIPT_FILE,
    USER_REQUIREMENTS_FILE,
    PREPROCESS_DATA_FILE,
    STATE_DICT_READY_FILE,
    STATE_DICT_FILE,
    GRADIENT_READY_FILE,
    GRADIENT_FILE,
    TRAINED_MODEL_FILE,
    CHECKPOINT_FILE,
    WAITING_PERIOD,
    ExampleDirs,
)
from tests.logger import testlogger
from tests.utils import (
    check_temp_dir_created,
    list_worker_nodes,
    leader_dir,
    worker_dir,
    has_worker_finished,
    leader_get_path,
    with_temp_dir,
    dynamic_import,
)


def run_node(
        role="WORKER",
        worker_count=1,
        dir_name="worker1",
        tensor_load=False,
):
    check_temp_dir_created()

    working_directory = TEMP_DIR / dir_name
    os.makedirs(working_directory / "output", exist_ok=True)

    argv = [
        "main.py",
        "--role", role,
        "--worker_count", str(worker_count)
    ]
    if tensor_load:
        argv.append("--tensor_load")

    with patch("sys.argv", argv):
        os.chdir(working_directory)
        from mcy_dist_ai.main import main
        return main()


class WatchLeader:
    def __init__(self, example_dir: Path, worker_count: int = 1):
        check_temp_dir_created()
        self.example_dir = example_dir
        self.worker_count = worker_count
        self.worker_nodes = list_worker_nodes(worker_count)
        self.state_dict_ready_path = leader_dir() / STATE_DICT_READY_FILE
        self.state_dict_path = leader_dir() / STATE_DICT_FILE
        if worker_count > 1:
            os.makedirs(leader_dir(), exist_ok=True)

    def send_file_to_leader(self, file: str):
        shutil.copy(
            self.example_dir / file,
            leader_dir() / file
        )

    async def wait_state_dict(self):
        while not os.path.exists(self.state_dict_ready_path):
            await asyncio.sleep(WAITING_PERIOD)
        if not os.path.exists(self.state_dict_path):
            raise FileNotFoundError(f"{self.state_dict_path} does not exist!")
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
            testlogger.info("Watch leader not started, because there's only one worker.")
            return

        testlogger.info("Watch leader started.")
        self.send_file_to_leader(USER_SCRIPT_FILE)
        self.send_file_to_leader(USER_REQUIREMENTS_FILE)
        while True:
            await self.wait_state_dict()

            for node in self.worker_nodes:
                self.send_state_dict_to_worker(node)

            if all(has_worker_finished(node) for node in self.worker_nodes):
                testlogger.info("Watch leader finished.")
                return


class WatchWorker:
    def __init__(self, example_dir: Path, node: str, worker_count: int = 1):
        check_temp_dir_created()
        self.example_dir = example_dir
        self.node = node
        self.worker_count = worker_count
        self.worker_dir = worker_dir(self.node)
        os.makedirs(self.worker_dir, exist_ok=True)

    def send_file_to_worker(self, file: str):
        shutil.copy(
            self.example_dir / file,
            self.worker_dir / file
        )

    def send_data_to_worker(self):
        data_dir = "data" if self.worker_count <= 1 else f"split_data/{self.node}"
        shutil.copytree(
            self.example_dir / data_dir,
            self.worker_dir / "data",
            dirs_exist_ok=True
        )

    def remove_checkpoint(self):
        if os.path.exists(self.worker_dir / CHECKPOINT_FILE):
            os.remove(self.worker_dir / CHECKPOINT_FILE)

    async def wait_gradient(self):
        while not os.path.exists(self.worker_dir / GRADIENT_READY_FILE):
            await asyncio.sleep(WAITING_PERIOD)
        if not os.path.exists(self.worker_dir / GRADIENT_FILE):
            raise FileNotFoundError(f"{self.worker_dir / GRADIENT_FILE} does not exist!")

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
        testlogger.info(f"Watch worker {self.node} started.")
        self.send_file_to_worker(USER_SCRIPT_FILE)
        self.send_file_to_worker(USER_REQUIREMENTS_FILE)
        self.send_data_to_worker()

        if self.worker_count == 1:
            # if there's only one worker, leader is not needed
            testlogger.info(f"Watch worker {self.node} finished.")
            return

        while True:
            await self.wait_gradient()

            if has_worker_finished(self.node):
                self.send_worker_finished_to_leader()

            self.send_gradient_to_leader()

            if has_worker_finished(self.node):
                testlogger.info(f"Watch worker {self.node} finished.")
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


def split_and_save_data_manually(split_into: int, example_dir: Path):
    if split_into > 1:
        preprocess_data = dynamic_import("preprocess_data", example_dir / PREPROCESS_DATA_FILE)
        preprocess_data.split_and_save_data(split_into=split_into, random_seed=42)


def split_and_save_data_by_mcy_script(split_into: int, example_dir: Path):
    from mcy_dist_ai.script.split_data import split_data
    shutil.rmtree(str(example_dir / "split_data"), ignore_errors=True)
    split_data(
        split_into,
        str(example_dir / "data"),
        str(example_dir / "split_data"),
        str(example_dir / USER_SCRIPT_FILE)
    )


def split_and_save_data(split_into: int, example_dir: Path, tensor_load: bool = False):
    if tensor_load:
        split_and_save_data_by_mcy_script(split_into, example_dir)
    else:
        split_and_save_data_manually(split_into, example_dir)


@with_temp_dir(clear_tmp_dir_end=False)
def train_model_parallel(
        worker_count: int, example_dir: Path = ExampleDirs.IMAGE_CLASSIFIER, tensor_load: bool = False
):
    workers = []

    split_and_save_data(worker_count, example_dir, tensor_load=tensor_load)

    for i in range(worker_count):
        workers.append(
            multiprocessing.Process(
                name=f"worker{i+1}",
                target=run_node,
                kwargs=dict(
                    role="WORKER",
                    worker_count=worker_count,
                    dir_name=f"worker{i+1}",
                    tensor_load=tensor_load,
                )
            )
        )

    p2p_network_simulator = multiprocessing.Process(
        name="p2p_network_simulator",
        target=simulate_p2p_network,
        kwargs=dict(
            example_dir=example_dir,
            worker_count=worker_count
        )
    )

    leader = multiprocessing.Process(
        name="leader",
        target=run_node,
        kwargs=dict(
            role="LEADER",
            worker_count=worker_count,
            dir_name="leader",
        )
    )

    [worker.start() for worker in workers]
    leader.start()
    p2p_network_simulator.start()
    [worker.join() for worker in workers]
    leader.join()
    p2p_network_simulator.join()


@with_temp_dir(clear_tmp_dir_end=False)
def train_model_sequential(
        worker_count: int, example_dir: Path = ExampleDirs.IMAGE_CLASSIFIER, tensor_load: bool = False
):
    split_and_save_data(worker_count, example_dir, tensor_load=tensor_load)

    for i in range(worker_count):
        watch_worker = WatchWorker(example_dir, "1", worker_count)
        watch_worker.remove_checkpoint()
        watch_worker.send_file_to_worker(USER_SCRIPT_FILE)
        watch_worker.send_file_to_worker(USER_REQUIREMENTS_FILE)
        watch_worker.node = str(i+1)
        watch_worker.send_data_to_worker()

        run_node(
            role="WORKER",
            worker_count=1,
            dir_name="worker1",
            tensor_load=tensor_load,
        )

        shutil.copy(
            watch_worker.worker_dir / "output" / TRAINED_MODEL_FILE,
            watch_worker.worker_dir / STATE_DICT_FILE
        )
