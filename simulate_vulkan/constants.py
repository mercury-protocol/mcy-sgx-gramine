from pathlib import Path
from pytorch.constants import USER_SCRIPT_FILE


WORKER_NODES_NUM = 2
LOCAL_USER_SCRIPT_PATH = Path(USER_SCRIPT_FILE)
LOCAL_SPLIT_DATA_PATH = Path("local_io/split_data")
LOCAL_DATA_PATH = Path("local_io/data")
