import torch
import torch.nn.functional as F
import torchvision

from pytorch.constants import DATA_PATH


BATCH_SIZE_TEST = 1000


test_data_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(DATA_PATH, train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=BATCH_SIZE_TEST, shuffle=True)


def test(network):
    global test_data_loader

    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_data_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data_loader.dataset),
        100. * correct / len(test_data_loader.dataset)))
