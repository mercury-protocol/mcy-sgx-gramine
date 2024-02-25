import os
from pathlib import Path


ROLE = None
WORKER_NODES_NUM = None

WORKER_ROLE = "WORKER"
LEADER_ROLE = "LEADER"

USER_SCRIPT_FILE = "user_script.py"
STATE_DICT_FILE = "state_dict.pth"
GRADIENT_FILE = "gradient.pth"
STATE_DICT_READY_FILE = "state_dict_ready.pth"
GRADIENT_READY_FILE = "gradient_ready.pth"
WORKER_FINISHED_FILE = "worker_finished.pth"
MONITOR_FILE = "monitor.pth"
TRAINED_MODEL_FILE = "trained_model.pth"
WATCHER_DATA_FILE = "data.pt" # in Vulkan watcher always saves data under this name

BASE_DIR = Path(os.getcwd())
OUTPUT_DIR = Path("/var/tmp/vulkan_trained_models")

DATA_PATH = BASE_DIR / "partition.pth"
USER_SCRIPT_PATH = BASE_DIR / USER_SCRIPT_FILE
AGGREGATED_STATE_DICT_PATH = BASE_DIR / STATE_DICT_FILE
TRAINED_MODEL_PATH = OUTPUT_DIR / TRAINED_MODEL_FILE

WAITING_PERIOD = 0.01
MONITORING_PERIOD = 10
LOG_INTERVAL = 50

def setup(role, worker_nodes_num):
    global ROLE, WORKER_NODES_NUM
    ROLE = role
    WORKER_NODES_NUM = worker_nodes_num