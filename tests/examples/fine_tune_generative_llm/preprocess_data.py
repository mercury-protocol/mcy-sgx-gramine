import os
import shutil

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer

from tests.examples.fine_tune_generative_llm.constants import (
    SPLIT_DATA_PATH,
    DATA_PATH,
    RAW_DATA_PATH,
    VALID_DATA_SPLIT_PARTITIONS
)


DATA = "mlabonne/guanaco-llama2-1k"
MODEL = "NousResearch/Llama-2-7b-chat-hf"


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def create_tokenized_datasets():
    dataset = load_dataset(DATA, cache_dir=RAW_DATA_PATH)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Manually postprocess tokenized_dataset to prepare it for training.
    # Remove the text column because the model does not accept raw text as an input:
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # Set the format of the dataset to return PyTorch tensors instead of lists:
    # this step is done in the create_data_loader() function in user_script.py

    # Create smaller subsets of the dataset to speed up the fine-tuning:
    tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

    return tokenized_datasets


def get_output_data_path(partition_num):
    return SPLIT_DATA_PATH / f"{partition_num}"


def save_tokenized_data():
    datasets = create_tokenized_datasets()
    datasets.save_to_disk(DATA_PATH)


def split_and_save_data(split_into=4, random_seed=42):
    if split_into not in VALID_DATA_SPLIT_PARTITIONS:
        raise Exception(f"Can only split yelp_review_full(1000) dataset into {VALID_DATA_SPLIT_PARTITIONS} partitions.")

    shutil.rmtree(SPLIT_DATA_PATH, ignore_errors=True)

    for i in range(split_into):
        os.makedirs(get_output_data_path(i + 1), exist_ok=True)

    if not os.path.exists(DATA_PATH):
        save_tokenized_data()
    tokenized_dataset = load_from_disk(str(DATA_PATH))

    train_dataset = tokenized_dataset["train"]
    train_dataset.shuffle(seed=random_seed)
    train_partition_length = len(train_dataset) // split_into

    for i in range(split_into):
        train_partition = train_dataset[i * train_partition_length:(i + 1) * train_partition_length]

        split_dataset = DatasetDict({
            "train": Dataset.from_dict(train_partition)
        })

        split_dataset.save_to_disk(get_output_data_path(i + 1))


if __name__ == "__main__":
    split_and_save_data(split_into=4)
