import evaluate
import torch
import torch.nn as nn
from pprint import pprint
from tests.examples.fine_tune_llm.constants import TRAINED_MODEL_PATH, DATA_PATH
from tests.examples.fine_tune_llm.user_script import (
    N_EPOCHS,
    device,
    create_data_loader,
    create_eval_data_loader,
    create_model,
    create_optimizer,
    create_extra_training_args,
    train_batch,
)


def train_model(data_path=DATA_PATH) -> nn.Module:
    data_loader = create_data_loader(data_path)
    model = create_model()
    optimizer = create_optimizer(model)
    extra_args = create_extra_training_args(data_loader, optimizer)

    print()
    print("training loop start")

    for epoch in range(N_EPOCHS):
        for batch in data_loader:
            train_batch(batch, model, optimizer, *extra_args)

    torch.save(model.state_dict(), TRAINED_MODEL_PATH)

    print()
    print("training loop finish")

    return model


def load_trained_model() -> nn.Module:
    state_dict = torch.load(TRAINED_MODEL_PATH)
    model = create_model()
    model.load_state_dict(state_dict)
    return model


def evaluate_model(model: nn.Module, data_path=DATA_PATH) -> float:
    eval_dataloader = create_eval_data_loader(data_path)
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    computed_metric = metric.compute()

    print()
    pprint(computed_metric)

    return float(computed_metric["accuracy"])


if __name__ == "__main__":
    # {'accuracy': 0.551}
    # trained_model = train_model()
    trained_model = load_trained_model()
    evaluate_model(trained_model)
