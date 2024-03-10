import os
import shutil
from functools import wraps
from unittest.mock import patch

from tests.constants import TEST_DIR, USER_SCRIPTS_DIR


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
                    user_script="image_classifier.py",
                    clear_tmp_dir_start=True, clear_tmp_dir_end=True):
    def decorator(func):
        @wraps(func)
        @patch("sys.argv", [
            "utils.py",
            "--role", role,
            "--worker_count", str(worker_count)
        ])
        def wrapper(*args, **kwargs):
            with TempDir(clear_tmp_dir_start=clear_tmp_dir_start, clear_tmp_dir_end=clear_tmp_dir_end) as tmp_dir:
                # to patch pytorch.constants.BASE_DIR
                with patch("os.getcwd", return_value=tmp_dir):
                    shutil.copy(f"{USER_SCRIPTS_DIR}/{user_script}", f"{tmp_dir}/user_script.py")
                    return func(*args, **kwargs)
        return wrapper
    return decorator

