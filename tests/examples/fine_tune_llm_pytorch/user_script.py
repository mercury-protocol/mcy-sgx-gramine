import evaluate
import torch
import torch.nn as nn

from datasets import load_from_disk
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, get_scheduler


N_EPOCHS = 1
LEARNING_RATE = 5e-5

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def create_eval_data_loader(path="data") -> DataLoader:
    tokenized_datasets = load_from_disk(path)
    return DataLoader(tokenized_datasets["test"], batch_size=8)


def create_data_loader(path="data") -> DataLoader:
    tokenized_datasets = load_from_disk(path)
    return DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8)


def create_model() -> nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased", num_labels=5
    )
    model.to(device)
    model.train()
    return model


def create_optimizer(model: nn.Module) -> Optimizer:
    return AdamW(model.parameters(), lr=LEARNING_RATE)


def create_args_from_data_loader(data_loader: DataLoader):
    num_training_steps = N_EPOCHS * len(data_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))

    return lr_scheduler, progress_bar


def train_batch(batch, model, optimizer, lr_scheduler, progress_bar):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)

    return loss  # for local logging purposes only


if __name__ == "__main__":
    # {'accuracy': 0.533}
    from pprint import pprint

    data_loader = create_data_loader()
    model = create_model()
    optimizer = create_optimizer(model)
    extra_args = create_args_from_data_loader(data_loader)

    print()
    print("training loop start")

    for epoch in range(N_EPOCHS):
        for batch in data_loader:
            train_batch(batch, model, optimizer, *extra_args)

    torch.save(model.state_dict(), "trained_reference_model.pth")

    print()
    print("training loop finish")

    eval_dataloader = create_eval_data_loader()
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
