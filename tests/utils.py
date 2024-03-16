import torch
import torch.nn.functional as F
import torchvision

from typing import Callable

from tests.constants import TEMP_OUTPUT_DIR


def load_model(create_model: Callable):
    model = create_model()
    model.load_state_dict(torch.load(TEMP_OUTPUT_DIR + "/trained_model.pth"))
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
