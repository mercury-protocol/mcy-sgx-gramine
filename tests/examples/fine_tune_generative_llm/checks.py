import torch
import torch.nn as nn
from transformers import pipeline
from tests.examples.fine_tune_generative_llm.constants import TRAINED_MODEL_PATH, DATA_PATH
from tests.examples.fine_tune_generative_llm.preprocess_data import create_tokenizer
from tests.examples.fine_tune_generative_llm.user_script import (
    N_EPOCHS,
    create_data_loader,
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


def evaluate_model(model: nn.Module, **kwargs) -> float:
    prompt = "What is a large language model?"
    tokenizer = create_tokenizer()
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])

    return 0.0


if __name__ == "__main__":
    train_model()
    # trained_model = load_trained_model()
    # evaluate_model(trained_model)
