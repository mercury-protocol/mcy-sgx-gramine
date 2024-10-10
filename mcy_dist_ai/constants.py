import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from mcy_dist_ai.logger import logger


LEADER_ROLE = "LEADER"
WORKER_ROLE = "WORKER"

parser = ArgumentParser()
parser.add_argument("--role", type=str, help="Node role - leader or worker")
parser.add_argument("--worker_count", type=int, help="Worker nodes count")
parser.add_argument(
    "--tensor_load",
    action='store_true',
    default=False,
    help="pass this arg when data was split by mcy script"
)
args = parser.parse_args()
if args.role is None:
    logger.error("Role argument is missing")
    sys.exit(1)
if args.role.upper() not in (LEADER_ROLE, WORKER_ROLE):
    logger.error(f"role must be {LEADER_ROLE} or {WORKER_ROLE}")
    sys.exit(1)
if args.role == LEADER_ROLE and args.worker_count is None:
    logger.error("Worker nodes count argument is required for leader")
    sys.exit(1)

ROLE = args.role.upper()
WORKER_NODES_NUM = int(args.worker_count)
if ROLE == LEADER_ROLE and WORKER_NODES_NUM == 1:
    logger.info("Leader is not starting because there's only one worker.")
    sys.exit(0)
TENSOR_LOAD = args.tensor_load


GRADIENT_FILE = "gradient.pth"
GRADIENT_READY_FILE = "gradient_ready.pth"
WORKER_FINISHED_FILE = "worker_finished.pth"

BASE_DIR = Path(os.getcwd())
OUTPUT_DIR = BASE_DIR / "output"

DATA_PATH = BASE_DIR / "data"
PARTITIONED_TENSORS_PATH = BASE_DIR / "partitioned_tensors"
USER_SCRIPT_PATH = BASE_DIR / "user_script.py"
STATE_DICT_READY_PATH = BASE_DIR / "state_dict_ready.pth"
STATE_DICT_PATH = BASE_DIR / "state_dict.pth"
TRAINED_MODEL_PATH = OUTPUT_DIR / "trained_model.pth"
MONITOR_PATH = BASE_DIR / "monitor.pth"
CHECKPOINT_PATH = BASE_DIR / "checkpoint.bin"


WAITING_PERIOD = 0.01
MONITORING_PERIOD = 10
LOG_INTERVAL = 50
