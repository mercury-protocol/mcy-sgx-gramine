import importlib
import os
import shutil
import torch

from pathlib import Path

from tests.constants import TEMP_DIR, WORKER_FINISHED_FILE, TRAINED_MODEL_FILE, USER_SCRIPT_FILE, CHECKS_FILE
from tests.exceptions import TempDirNotCreated


class TempDir:
    def __init__(self, subdir_name=None, clear_tmp_dir_start=True, clear_tmp_dir_end=True):
        self.clear_tmp_dir_start = clear_tmp_dir_start
        self.clear_tmp_dir_end = clear_tmp_dir_end
        self.tmp_dir = TEMP_DIR / subdir_name if subdir_name else TEMP_DIR

    def __enter__(self):
        if self.clear_tmp_dir_start:
            try:
                shutil.rmtree(self.tmp_dir)
            except FileNotFoundError:
                pass
        os.makedirs(self.tmp_dir, exist_ok=True)
        return self.tmp_dir

    def __exit__(self, *args, **kwargs):
        if self.clear_tmp_dir_end:
            shutil.rmtree(self.tmp_dir)


def with_temp_dir(clear_tmp_dir_start=True, clear_tmp_dir_end=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TempDir(
                    subdir_name=None,
                    clear_tmp_dir_start=clear_tmp_dir_start,
                    clear_tmp_dir_end=clear_tmp_dir_end
            ):
                func(*args, **kwargs)
        return wrapper
    return decorator


def check_temp_dir_created():
    if not os.path.exists(TEMP_DIR):
        raise TempDirNotCreated(f"{TEMP_DIR} has not been created. Consider using with_temp_dir or TempDir.")


def leader_dir() -> Path:
    return TEMP_DIR / "leader"


def leader_get_path(worker_node: str, file: str) -> Path:
    if "." in file:
        name, extension = file.split(".")
        file = f"{name}_{worker_node}.{extension}"
    else:
        file = f"{file}_{worker_node}"
    return leader_dir() / file


def worker_dir(node: str) -> Path:
    return TEMP_DIR / f"worker{node}"


def has_worker_finished(node: str) -> bool:
    return os.path.exists(worker_dir(node) / WORKER_FINISHED_FILE)


def list_worker_nodes(worker_count: int) -> list[str]:
    return [str(i + 1) for i in range(worker_count)]


def dynamic_import(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, os.path.abspath(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(model_dir: Path, example_dir: Path):
    user_script = dynamic_import("user_script", example_dir / USER_SCRIPT_FILE)
    model = user_script.create_model()
    model.load_state_dict(torch.load(model_dir / TRAINED_MODEL_FILE))
    return model


def evaluate_model(model: torch.nn.Module, example_dir: Path) -> float:
    checks = dynamic_import("checks", example_dir / CHECKS_FILE)
    data_path = example_dir / "data"
    accuracy = checks.evaluate_model(model, data_path)
    return accuracy
