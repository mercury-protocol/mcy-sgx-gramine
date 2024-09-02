"""This file is for doing manual tests, experimenting and debugging"""
from tests.constants import ExampleDirs
from tests.simulation import train_model_parallel
from tests.utils import load_model, evaluate_model, get_model_dir


if __name__ == "__main__":
    example_dir = ExampleDirs.FINE_TUNE_LLM_PYTORCH
    worker_count = 2

    train_model_parallel(worker_count, example_dir)
    model = load_model(get_model_dir(worker_count), example_dir)
    model_accuracy = evaluate_model(model, example_dir)
