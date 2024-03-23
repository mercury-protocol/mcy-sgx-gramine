import os
import shutil
from git import Repo, GitCommandError
from functools import wraps
from unittest.mock import patch

from tests.constants import TEMP_DIR, EXAMPLES_DIR, REPO_DIR, CONSTANTS_PATCH


class TempDir:
    def __init__(self, clear_tmp_dir_start=True, clear_tmp_dir_end=True):
        self.clear_tmp_dir_start = clear_tmp_dir_start
        self.clear_tmp_dir_end = clear_tmp_dir_end
        self.tmp_dir = TEMP_DIR

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


class ApplyPatch:
    def __init__(self, patch_file=CONSTANTS_PATCH):
        self.patch_file = patch_file
        self.repo = Repo(REPO_DIR)

    def apply(self):
        self.repo.git.apply(self.patch_file)

    def reverse_apply(self):
        self.repo.git.apply("--reverse", self.patch_file)

    def __enter__(self):
        try:
            self.apply()
        except GitCommandError:
            self.reverse_apply()
            self.apply()

    def __exit__(self, *args, **kwargs):
        self.reverse_apply()


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
                with ApplyPatch(patch_file=CONSTANTS_PATCH):
                    os.makedirs(f"{tmp_dir}/output", exist_ok=True)
                    shutil.copy(f"{example_dir}/user_script.py", f"{tmp_dir}/user_script.py")
                    if os.path.exists(f"{example_dir}/data"):
                        shutil.copytree(f"{example_dir}/data", f"{tmp_dir}/data")

                    return func(*args, **kwargs)
        return wrapper
    return decorator
