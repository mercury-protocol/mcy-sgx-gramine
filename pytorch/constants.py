import os
from pathlib import Path


ROLE = os.getenv("ROLE")
WORKER_NODES_NUM = int(os.getenv("WORKER_NODES_NUM", 0))

WORKER_ROLE = "WORKER"
LEADER_ROLE = "LEADER"

USER_SCRIPT_FILE = "user_script.py"
DATA_DIR = "data"
STATE_DICT_FILE = "state_dict.pth"
GRADIENT_FILE = "gradient.pth"
TRAINING_COMPLETE_FILE = "training_complete"
BATCH_AGGREGATION_COMPLETE_FILE = "batch_aggregation_complete"
MONITOR_FILE = "monitor"

IO_DIR = Path("../io")
LEADER_DIR = IO_DIR / "leader"
WORKER_DIR = IO_DIR / "worker"
WATCHER_DIR = IO_DIR / "watcher"

AGGREGATED_STATE_DICT_PATH = LEADER_DIR / STATE_DICT_FILE

WAITING_PERIOD = 0.01
MONITOR_PERIOD = 10
