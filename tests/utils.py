import os
import shutil
import torch
import torch.nn.functional as F
import torchvision

from pathlib import Path
from typing import Callable
from unittest.mock import patch

from tests.constants import TEMP_DIR
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


def run_node(
        role="WORKER",
        worker_count=1,
        dir_name="worker1",
):
    check_temp_dir_created()

    working_directory = TEMP_DIR / dir_name
    os.makedirs(working_directory / "output", exist_ok=True)
    with patch("sys.argv", [
        "main.py",
        "--role", role,
        "--worker_count", str(worker_count)
    ]):
        os.chdir(working_directory)
        from pytorch.main import main
        return main()


def load_model(path: Path, create_model: Callable):
    model = create_model()
    model.load_state_dict(torch.load(path / "trained_model.pth"))
    return model


def evaluate_model(model, data_path, batch_size=1000):
    test_data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_path, train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_data_loader.dataset)
    accuracy = correct / len(test_data_loader.dataset)
    print(f"\nTest set: Avg. loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_data_loader.dataset)} ({100. * accuracy:.0f}%)\n")

    return accuracy
