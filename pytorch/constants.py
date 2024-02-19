import os
from pathlib import Path


ROLE = os.getenv("ROLE")
WORKER_NODES_NUM = int(os.getenv("WORKER_NODES_NUM", 0))

WORKER_ROLE = "WORKER"
LEADER_ROLE = "LEADER"

USER_SCRIPT_FILE = "user_script.py"
STATE_DICT_FILE = "state_dict.pth"
GRADIENT_FILE = "gradient.pth"
STATE_DICT_READY_FILE = "state_dict_ready.pth"
GRADIENT_READY_FILE = "gradient_ready.pth"
WORKER_FINISHED_FILE = "worker_finished.pth"
MONITOR_FILE = "monitor.pth"

IO_DIR = Path("../io")
LEADER_DIR = IO_DIR / "leader"
WORKER_DIR = IO_DIR / "worker"

DATA_PATH = WORKER_DIR / "data"
USER_SCRIPT_PATH = (LEADER_DIR if ROLE == LEADER_ROLE else WORKER_DIR) / USER_SCRIPT_FILE
AGGREGATED_STATE_DICT_PATH = LEADER_DIR / STATE_DICT_FILE

WAITING_PERIOD = 0.01
MONITORING_PERIOD = 10
LOG_INTERVAL = 50
