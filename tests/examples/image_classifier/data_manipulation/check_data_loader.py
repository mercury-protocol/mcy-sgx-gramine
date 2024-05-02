import random
import matplotlib.pyplot as plt

from tests.examples.image_classifier.data_manipulation.constants import SPLIT_DATA_PATH
from tests.examples.image_classifier.user_script import create_data_loader


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


if __name__ == "__main__":
    data_loader = create_data_loader(SPLIT_DATA_PATH + "/1")
    check_data(data_loader)
