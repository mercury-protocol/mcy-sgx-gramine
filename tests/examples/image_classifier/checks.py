import random
import matplotlib.pyplot as plt
from tests.examples.image_classifier.constants import DATA_PATH, SPLIT_DATA_PATH
from tests.examples.image_classifier.user_script import *
from tests.utils import evaluate_model  # TODO: define this here or use a universal one


def check_data():
    data_loader = create_data_loader(SPLIT_DATA_PATH / "1")
    examples = list(enumerate(data_loader))
    batch_idx, (example_data, example_targets) = random.choice(examples)

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()


def check_training():
    data_loader = create_data_loader(DATA_PATH)
    model = create_model()
    optimizer = create_optimizer(model)
    for epoch in range(N_EPOCHS):
        for batch_idx, batch in enumerate(data_loader):
            train_batch(batch, model, optimizer)

    evaluate_model(model, DATA_PATH)


if __name__ == "__main__":
    check_data()

    # Test set: Avg. loss: 0.1986, Accuracy: 9410/10000 (94%)
    check_training()
