import os
import shutil
from git import Repo, GitCommandError
from functools import wraps
from unittest.mock import patch

from tests.constants import TEMP_DIR, EXAMPLES_DIR, REPO_DIR, CONSTANTS_PATCH


class TempDirs:
    def __init__(self, dirs_num=1, clear_tmp_dirs_start=True, clear_tmp_dirs_end=True):
        self.clear_tmp_dirs_start = clear_tmp_dirs_start
        self.clear_tmp_dirs_end = clear_tmp_dirs_end
        self.tmp_dirs = [f"{TEMP_DIR}_{i + 1}" if i else TEMP_DIR for i in range(dirs_num)]

    def make_dirs(self):
        for tmp_dir in self.tmp_dirs:
            os.makedirs(tmp_dir, exist_ok=True)

    def remove_dirs(self):
        for tmp_dir in self.tmp_dirs:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def __enter__(self):
        if self.clear_tmp_dirs_start:
            self.remove_dirs()
        self.make_dirs()
        return self.tmp_dirs

    def __exit__(self, *args, **kwargs):
        if self.clear_tmp_dirs_end:
            self.remove_dirs()


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
                    clear_tmp_dirs_start=True, clear_tmp_dirs_end=True):

    example_dir = f"{EXAMPLES_DIR}/{example_dir}"

    def decorator(func):
        @wraps(func)
        @patch("sys.argv", [
            "main.py",
            "--role", role,
            "--worker_count", str(worker_count)
        ])
        def wrapper(*args, **kwargs):
            with TempDirs(
                    dirs_num=worker_count,
                    clear_tmp_dirs_start=clear_tmp_dirs_start,
                    clear_tmp_dirs_end=clear_tmp_dirs_end
            ) as tmp_dirs:
                for tmp_dir in tmp_dirs:
                    os.makedirs(f"{tmp_dir}/output", exist_ok=True)
                    shutil.copy(f"{example_dir}/user_script.py", f"{tmp_dir}/user_script.py")
                    if os.path.exists(f"{example_dir}/data"):
                        shutil.copytree(f"{example_dir}/data", f"{tmp_dir}/data")

                with ApplyPatch(patch_file=CONSTANTS_PATCH):
                    return func(*args, **kwargs)

        return wrapper

    return decorator
