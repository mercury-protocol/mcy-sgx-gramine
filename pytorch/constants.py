import os
from pathlib import Path


ROLE = None
WORKER_NODES_NUM = None

LEADER_ROLE = "LEADER"
WORKER_ROLE = "WORKER"
WORKER_LLM_ROLE = "WORKER-LLM"

GRADIENT_FILE = "gradient.pth"
GRADIENT_READY_FILE = "gradient_ready.pth"
WORKER_FINISHED_FILE = "worker_finished.pth"

BASE_DIR = Path(os.getcwd())
OUTPUT_DIR = Path("/var/tmp/vulkan_trained_models")

DATA_PATH = BASE_DIR / "partition.pth"
USER_SCRIPT_PATH = BASE_DIR / "user_script.py"
STATE_DICT_READY_PATH = BASE_DIR / "state_dict_ready.pth"
STATE_DICT_PATH = BASE_DIR / "state_dict.pth"
TRAINED_MODEL_PATH = OUTPUT_DIR / "trained_model.pth"
MONITOR_PATH = BASE_DIR / "monitor.pth"
CHECKPOINT_PATH = BASE_DIR / "checkpoint.bin"
WATCHER_DATA_PATH = BASE_DIR / "data.pt"

WAITING_PERIOD = 0.01
MONITORING_PERIOD = 10
LOG_INTERVAL = 50


def set_role_and_worker_node_num(role: str, worker_nodes_num: int):
    global ROLE, WORKER_NODES_NUM
    ROLE = role.upper()
    WORKER_NODES_NUM = worker_nodes_num
