import os
import shutil
import sys
from functools import wraps
from unittest.mock import patch

from tests.constants import TEST_DIR, EXAMPLES_DIR


class TempDir:
    def __init__(self, clear_tmp_dir_start=True, clear_tmp_dir_end=True):
        self.clear_tmp_dir_start = clear_tmp_dir_start
        self.clear_tmp_dir_end = clear_tmp_dir_end
        self.tmp_dir = TEST_DIR + "/temp"

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
                    example_dir="image_classifier",
                    clear_tmp_dir_start=True, clear_tmp_dir_end=True):

    example_dir = f"{EXAMPLES_DIR}/{example_dir}"

    def decorator(func):
        @wraps(func)
        @patch("sys.argv", [
            "main.py",
            "--role", role,
            "--worker_count", str(worker_count)
        ])
        def wrapper(*args, **kwargs):
            with TempDir(clear_tmp_dir_start=clear_tmp_dir_start, clear_tmp_dir_end=clear_tmp_dir_end) as tmp_dir:
                # to patch pytorch.constants.BASE_DIR
                with patch("os.getcwd", return_value=tmp_dir):
                    shutil.copy(f"{example_dir}/user_script.py", f"{tmp_dir}/user_script.py")
                    if os.path.exists(f"{example_dir}/data"):
                        shutil.copytree(f"{example_dir}/data", f"{tmp_dir}/data")
                    else:
                        sys.exit(f"No data folder in {example_dir}.")

                    return func(*args, **kwargs)
        return wrapper
    return decorator

