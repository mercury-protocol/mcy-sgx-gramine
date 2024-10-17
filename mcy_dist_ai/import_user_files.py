import importlib
import os
import subprocess
import sys
import types
from pathlib import Path
from time import sleep

from mcy_dist_ai.constants import (
    WAITING_PERIOD,
    USER_SCRIPT_PATH,
    USER_REQUIREMENTS_PATH,
)
from mcy_dist_ai.logger import logger


def import_user_script(user_script_path: str) -> types.ModuleType:
    if not os.path.exists(user_script_path):
        raise FileNotFoundError("Invalid path specified for user_script.py")

    spec = importlib.util.spec_from_file_location("user_script", user_script_path)
    user_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_script)
    logger.info("user_script.py imported.")
    return user_script


def wait_and_import_user_script() -> types.ModuleType:
    abs_script_path = wait_file_transfer_complete(USER_SCRIPT_PATH)
    return import_user_script(abs_script_path)


def wait_and_install_user_requirements():
    wait_file_transfer_complete(USER_REQUIREMENTS_PATH)

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "user_requirements.txt"])
        logger.info("user_requirements.txt installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred while installing user_requirements.txt: {e}")
        sys.exit(1)


def wait_file_transfer_complete(path: Path) -> str:
    abs_path = os.path.abspath(path)
    while not os.path.exists(abs_path):
        sleep(WAITING_PERIOD)

    sleep(2)  # TODO: use a safer method to wait if file is completely copied
    return abs_path
