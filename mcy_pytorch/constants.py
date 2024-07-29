import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from mcy_pytorch.logger import logger


LEADER_ROLE = "LEADER"
WORKER_ROLE = "WORKER"
LEADER_PRESET_ROLE = "LEADER-PRESET"
WORKER_PRESET_ROLE = "WORKER-PRESET"
WORKER_LLM_ROLE = "WORKER-LLM"
SUPPORTED_PRESET_TASKS = ("llm_classification",)

parser = ArgumentParser()
parser.add_argument("--role", type=str, help="Node role - leader, worker or worker-llm")
parser.add_argument("--worker_count", type=int, help="Worker nodes count")
parser.add_argument("--data", type=str, help="Data path in huggingface for preset modes")
parser.add_argument("--model", type=str, help="Model path in huggingface for preset modes")
parser.add_argument("--task", type=str, help="Training task to do in preset modes")
args = parser.parse_args()
if args.role is None:
    logger.error("Role argument is missing.")
    sys.exit(1)
if args.role.upper() not in (LEADER_ROLE, WORKER_ROLE, LEADER_PRESET_ROLE, WORKER_PRESET_ROLE, WORKER_LLM_ROLE):
    logger.error(f"Role must be {LEADER_ROLE}, {WORKER_ROLE}, {LEADER_PRESET_ROLE}, {WORKER_PRESET_ROLE} "
                 f"or {WORKER_LLM_ROLE}.")
    sys.exit(1)
if args.role == LEADER_ROLE and args.worker_count is None:
    logger.error("Worker nodes count argument is required for leader.")
    sys.exit(1)
if args.role in (LEADER_PRESET_ROLE, WORKER_PRESET_ROLE):
    if not args.data:
        logger.error("Data path must be specified.")
        sys.exit(1)
    elif not args.model:
        logger.error("Model path must be specified.")
        sys.exit(1)
    elif args.task.lower() not in SUPPORTED_PRESET_TASKS:
        logger.error(f"Only {SUPPORTED_PRESET_TASKS} tasks are supported.")
        sys.exit(1)

ROLE = args.role.upper()
WORKER_NODES_NUM = int(args.worker_count)
DATA = args.data
MODEL = args.model
TASK = args.task.lower()
if ROLE == LEADER_ROLE and WORKER_NODES_NUM == 1:
    logger.info("Leader is not starting because there's only one worker.")
    sys.exit(0)


GRADIENT_FILE = "gradient.pth"
GRADIENT_READY_FILE = "gradient_ready.pth"
WORKER_FINISHED_FILE = "worker_finished.pth"

BASE_DIR = Path(os.getcwd())
OUTPUT_DIR = BASE_DIR / "output"

DATA_PATH = BASE_DIR / "data"
USER_SCRIPT_PATH = (
        BASE_DIR / "user_script.py" if ROLE not in (LEADER_PRESET_ROLE, WORKER_PRESET_ROLE)
        else BASE_DIR / "preset_user_scripts" / TASK / "user_script.py"
)
STATE_DICT_READY_PATH = BASE_DIR / "state_dict_ready.pth"
STATE_DICT_PATH = BASE_DIR / "state_dict.pth"
TRAINED_MODEL_PATH = OUTPUT_DIR / "trained_model.pth"
MONITOR_PATH = BASE_DIR / "monitor.pth"
CHECKPOINT_PATH = BASE_DIR / "checkpoint.bin"
WATCHER_DATA_PATH = BASE_DIR / "data.pt"


WAITING_PERIOD = 0.01
MONITORING_PERIOD = 10
LOG_INTERVAL = 50
