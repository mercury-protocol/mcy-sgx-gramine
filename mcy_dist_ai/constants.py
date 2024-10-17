import os
from pathlib import Path


LEADER_ROLE = "LEADER"
WORKER_ROLE = "WORKER"

GRADIENT_FILE = "gradient.pth"
GRADIENT_READY_FILE = "gradient_ready.pth"
WORKER_FINISHED_FILE = "worker_finished.pth"

BASE_DIR = Path(os.getcwd())
OUTPUT_DIR = BASE_DIR / "output"

DATA_PATH = BASE_DIR / "data"
PARTITIONED_TENSORS_PATH = BASE_DIR / "partitioned_tensors"
USER_SCRIPT_PATH = BASE_DIR / "user_script.py"
USER_REQUIREMENTS_PATH = BASE_DIR / "user_requirements.txt"
STATE_DICT_READY_PATH = BASE_DIR / "state_dict_ready.pth"
STATE_DICT_PATH = BASE_DIR / "state_dict.pth"
TRAINED_MODEL_PATH = OUTPUT_DIR / "trained_model.pth"
MONITOR_PATH = BASE_DIR / "monitor.pth"
CHECKPOINT_PATH = BASE_DIR / "checkpoint.bin"


WAITING_PERIOD = 0.01
MONITORING_PERIOD = 10
LOG_INTERVAL = 50
