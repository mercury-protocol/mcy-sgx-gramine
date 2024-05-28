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


def create_data_loader(path) -> DataLoader:
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


def create_extra_training_args(data_loader: DataLoader, optimizer: Optimizer):
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


# ---------- for testing purpose ----------
def create_eval_data_loader(path) -> DataLoader:
    tokenized_datasets = load_from_disk(path)
    return DataLoader(tokenized_datasets["test"], batch_size=8)
