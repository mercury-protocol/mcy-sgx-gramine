import os
import shutil
from functools import wraps
from unittest.mock import patch

from tests.constants import TEMP_DIR, EXAMPLES_DIR


class TempDir:
    def __init__(self, dir_name, clear_tmp_dir_start=True, clear_tmp_dir_end=True):
        self.clear_tmp_dir_start = clear_tmp_dir_start
        self.clear_tmp_dir_end = clear_tmp_dir_end
        self.tmp_dir = TEMP_DIR / dir_name

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


def pytorch_context(role="WORKER", worker_count=1,
                    temp_dir_name="worker1",
                    example_dir="image_classifier",
                    clear_tmp_dir_start=True, clear_tmp_dir_end=True):

    example_dir = EXAMPLES_DIR / example_dir

    def decorator(func):
        @wraps(func)
        @patch("sys.argv", [
            "main.py",
            "--role", role,
            "--worker_count", str(worker_count)
        ])
        def wrapper(*args, **kwargs):
            with TempDir(
                    temp_dir_name,
                    clear_tmp_dir_start=clear_tmp_dir_start,
                    clear_tmp_dir_end=clear_tmp_dir_end
            ) as tmp_dir:
                os.makedirs(tmp_dir / "output", exist_ok=True)
                shutil.copy(example_dir / "user_script.py", tmp_dir / "user_script.py")
                if os.path.exists(example_dir / "data"):
                    shutil.copytree(example_dir / "data", tmp_dir / "data")

                with patch("os.getcwd", return_value=tmp_dir):
                    return func(*args, **kwargs)
        return wrapper
    return decorator
