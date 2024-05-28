import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F
from tests.examples.image_classifier.constants import DATA_PATH, SPLIT_DATA_PATH
from tests.examples.image_classifier.user_script import (
    N_EPOCHS,
    create_data_loader,
    create_eval_data_loader,
    create_model,
    create_optimizer,
    train_batch,
)


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


def train_model(data_path=DATA_PATH):
    data_loader = create_data_loader(data_path)
    model = create_model()
    optimizer = create_optimizer(model)
    for epoch in range(N_EPOCHS):
        for batch_idx, batch in enumerate(data_loader):
            train_batch(batch, model, optimizer)

    return model


def evaluate_model(model, data_path=DATA_PATH) -> float:
    eval_data_loader = create_eval_data_loader(data_path)

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in eval_data_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(eval_data_loader.dataset)
    accuracy = correct / len(eval_data_loader.dataset)
    print(f"\nTest set: Avg. loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(eval_data_loader.dataset)} ({100. * accuracy:.0f}%)\n")

    return float(accuracy)


if __name__ == "__main__":
    check_data()

    # Test set: Avg. loss: 0.1986, Accuracy: 9410/10000 (94%)
    trained_model = train_model()
    evaluate_model(trained_model, DATA_PATH)
