import os
import shutil
import torch
import torch.nn.functional as F
import torchvision

from pathlib import Path
from typing import Callable
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


def run_node(
        role="WORKER",
        worker_count=1,
        temp_dir_name="worker1",
        example_dir="image_classifier",
        clear_tmp_dir_start=True,
        clear_tmp_dir_end=True
):

    example_dir = EXAMPLES_DIR / example_dir

    with TempDir(
            temp_dir_name,
            clear_tmp_dir_start=clear_tmp_dir_start,
            clear_tmp_dir_end=clear_tmp_dir_end
    ) as tmp_dir:
        os.makedirs(tmp_dir / "output", exist_ok=True)
        shutil.copy(example_dir / "user_script.py", tmp_dir / "user_script.py")
        if os.path.exists(example_dir / "data"):
            shutil.copytree(example_dir / "data", tmp_dir / "data")

        with patch("sys.argv", [
            "main.py",
            "--role", role,
            "--worker_count", str(worker_count)
        ]):
            os.chdir(tmp_dir)
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
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data_loader.dataset),
        100. * correct / len(test_data_loader.dataset)))
