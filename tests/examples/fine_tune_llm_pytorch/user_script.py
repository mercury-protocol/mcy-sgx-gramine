import evaluate
import torch
import torch.nn as nn

from datasets import load_dataset
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler


N_EPOCHS = 1
LEARNING_RATE = 5e-5

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def create_tokenized_datasets():
    dataset = load_dataset("yelp_review_full", cache_dir="data")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Next, manually postprocess tokenized_dataset to prepare it for training.
    # 1. Remove the text column because the model does not accept raw text as an input:
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # 2. Rename the label column to labels because the model expects the argument to be named labels:
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # 3. Set the format of the dataset to return PyTorch tensors instead of lists:
    tokenized_datasets.set_format("torch")

    return tokenized_datasets


def create_eval_data_loader() -> DataLoader:
    tokenized_datasets = create_tokenized_datasets()
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    return DataLoader(small_eval_dataset, batch_size=8)


def create_data_loader() -> DataLoader:
    tokenized_datasets = create_tokenized_datasets()

    # Create a smaller subset of the dataset to speed up the fine-tuning:
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

    return DataLoader(small_train_dataset, shuffle=True, batch_size=8)


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
