import evaluate
import torch

from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler


N_EPOCHS = 1
LEARNING_RATE = 5e-5


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


dataset = load_dataset("yelp_review_full", cache_dir="data")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Next, manually postprocess tokenized_dataset to prepare it for training.
# 1. Remove the text column because the model does not accept raw text as an input:
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# 2. Rename the label column to labels because the model expects the argument to be named labels:
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# 3. Set the format of the dataset to return PyTorch tensors instead of lists:
tokenized_datasets.set_format("torch")

# Then create a smaller subset of the dataset to speed up the fine-tuning:
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", num_labels=5
)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = N_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(N_EPOCHS):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


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


if __name__ == "__main__":
    from pprint import pprint
    print()
    pprint(computed_metric)
