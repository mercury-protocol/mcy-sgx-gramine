import torch
import torch.nn as nn

from datasets import load_from_disk
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, get_scheduler, BitsAndBytesConfig

from tests.examples.fine_tune_generative_llm.constants import OPTIMIZER_MAPPING


DATA = "mlabonne/guanaco-llama2-1k"
MODEL = "google-bert/bert-base-cased"  # "NousResearch/Llama-2-7b-chat-hf"
OPTIMIZER = "paged_adamw_32bit"

N_EPOCHS = 1
LEARNING_RATE = 2e-4

IS_CUDA_AVAILABLE = torch.cuda.is_available()
USE_4BIT = True  # Activate 4-bit precision base model loading
BNB_4BIT_COMPUTE_DTYPE = "float16"  # Compute dtype for 4-bit base models
BNB_4BIT_QUANT_TYPE = "nf4"  # Quantization type (fp4 or nf4)
USE_NESTED_QUANT = False  # Activate nested quantization for 4-bit base models (double quantization)
COMPUTE_DTYPE = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)  # Load tokenizer and model with QLoRA configuration
DEVICE_MAP = {"": 0} if IS_CUDA_AVAILABLE else None  # Load the entire model on the GPU 0

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=USE_NESTED_QUANT,
) if IS_CUDA_AVAILABLE else None


def create_data_loader(path) -> DataLoader:
    tokenized_datasets = load_from_disk(path)
    tokenized_datasets.set_format("torch")
    return DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8)


def create_model() -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb_config,
        device_map=DEVICE_MAP
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.train()
    return model


def create_optimizer(model: nn.Module) -> Optimizer:
    optimizer_class = OPTIMIZER_MAPPING.get(OPTIMIZER)
    if not optimizer_class:
        raise ValueError(f"Optimizer {OPTIMIZER} is not supported")

    return optimizer_class(model.parameters(), lr=LEARNING_RATE)


def create_extra_training_args(data_loader: DataLoader, optimizer: Optimizer):
    num_training_steps = N_EPOCHS * len(data_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))

    return lr_scheduler, progress_bar


def train_batch(batch, model, optimizer, lr_scheduler, progress_bar):
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    progress_bar.update(1)

    return loss  # for local logging purposes only
