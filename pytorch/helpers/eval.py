import random
import torch
import matplotlib.pyplot as plt


def check_data(test_loader):
    examples = list(enumerate(test_loader))
    batch_idx, (example_data, example_targets) = random.choice(examples)

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()


def make_predictions(network, test_loader):
    examples = list(enumerate(test_loader))
    batch_idx, (example_data, example_targets) = random.choice(examples)

    with torch.no_grad():
        output = network(example_data)

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Prediction: {output.data.max(1, keepdim=True)[1][i].item()}\n"
                  f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()
