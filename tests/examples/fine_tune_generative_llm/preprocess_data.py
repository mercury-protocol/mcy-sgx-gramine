from datasets import load_dataset
from transformers import AutoTokenizer

from tests.examples.fine_tune_generative_llm.constants import (
    SPLIT_DATA_PATH,
    DATA_PATH,
    RAW_DATA_PATH,
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

    # Next, manually postprocess tokenized_dataset to prepare it for training.
    # 1. Remove the text column because the model does not accept raw text as an input:
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # 2. Rename the label column to labels because the model expects the argument to be named labels:
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # 3. Set the format of the dataset to return PyTorch tensors instead of lists:
    # this step is done in the create_data_loader() function in user_script.py

    # Create smaller subsets of the dataset to speed up the fine-tuning:
    tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    tokenized_datasets["test"] = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    return tokenized_datasets


def get_output_data_path(partition_num):
    return SPLIT_DATA_PATH / f"{partition_num}"


def save_tokenized_data():
    datasets = create_tokenized_datasets()
    datasets.save_to_disk(DATA_PATH)


if __name__ == "__main__":
    save_tokenized_data()
