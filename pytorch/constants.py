from pathlib import Path


STATE_DICT_FILE = "state_dict.pth"
OPTIMIZER_FILE = "optimizer.pth"
GRADIENT_FILE = "gradient.pth"
TRAINING_COMPLETE_FILE = "training_complete"
BATCH_AGGREGATION_COMPLETE_FILE = "batch_aggregation_complete"

IO_DIR = Path("io")
LEADER_DIR = IO_DIR / "leader"
WORKER_DIR = IO_DIR / "worker"
WATCHER_DIR = IO_DIR / "watcher"

SPLIT_DATA_PATH = IO_DIR / "split_data"
DATA_PATH = IO_DIR / "data"

AGGREGATED_STATE_DICT_PATH = LEADER_DIR / STATE_DICT_FILE

WAITING_PERIOD = 0.01
