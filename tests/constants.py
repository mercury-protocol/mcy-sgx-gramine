import os
from pathlib import Path


TEST_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = TEST_DIR / "temp"
EXAMPLES_DIR = TEST_DIR / "examples"

WORKER_FINISHED_FILE = "worker_finished.pth"
USER_SCRIPT_FILE = "user_script.py"
USER_REQUIREMENTS_FILE = "user_requirements.txt"
CHECKS_FILE = "checks.py"
PREPROCESS_DATA_FILE = "preprocess_data.py"
STATE_DICT_READY_FILE = "state_dict_ready.pth"
STATE_DICT_FILE = "state_dict.pth"
GRADIENT_READY_FILE = "gradient_ready.pth"
GRADIENT_FILE = "gradient.pth"
TRAINED_MODEL_FILE = "trained_model.pth"
CHECKPOINT_FILE = "checkpoint.bin"

WAITING_PERIOD = 0.01


class ExampleDirs:
    FINE_TUNE_LLM = EXAMPLES_DIR / "fine_tune_llm"
    IMAGE_CLASSIFIER = EXAMPLES_DIR / "image_classifier"
