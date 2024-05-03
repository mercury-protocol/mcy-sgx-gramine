import multiprocessing

from tests.constants import TEMP_DIR, ExampleDirs
from tests.examples.image_classifier.user_script import create_model
from tests.simulation import (
    run_node,
    simulate_p2p_network,
    train_image_classifier_parallel,
    train_image_classifier_sequential,
)
from tests.utils import load_model, evaluate_model, with_temp_dir


def test_train_image_classifier_one_worker_parallel():
    train_image_classifier_parallel(worker_count=1)

    model = load_model(TEMP_DIR / "worker1/output", create_model)
    model_accuracy = evaluate_model(model, ExampleDirs.IMAGE_CLASSIFIER / "data")
    assert model_accuracy > 0.94


def test_train_image_classifier_two_workers_parallel():
    train_image_classifier_parallel(worker_count=2)

    model = load_model(TEMP_DIR / "leader/output", create_model)
    model_accuracy = evaluate_model(model, ExampleDirs.IMAGE_CLASSIFIER / "data")
    assert model_accuracy > 0.94  # TODO: FAIL: accuracy reduced to 0.91


def test_train_image_classifier_four_workers_parallel():
    train_image_classifier_parallel(worker_count=4)

    model = load_model(TEMP_DIR / "leader/output", create_model)
    model_accuracy = evaluate_model(model, ExampleDirs.IMAGE_CLASSIFIER / "data")
    assert model_accuracy > 0.94  # TODO: FAIL: accuracy reduced to 0.84!!


def test_train_image_classifier_one_worker_sequential():
    train_image_classifier_sequential(worker_count=1)

    model = load_model(TEMP_DIR / "worker1/output", create_model)
    model_accuracy = evaluate_model(model, ExampleDirs.IMAGE_CLASSIFIER / "data")
    assert model_accuracy > 0.94


def test_train_image_classifier_two_workers_sequential():
    train_image_classifier_sequential(worker_count=2)

    model = load_model(TEMP_DIR / "worker1/output", create_model)
    model_accuracy = evaluate_model(model, ExampleDirs.IMAGE_CLASSIFIER / "data")
    assert model_accuracy > 0.94


def test_train_image_classifier_four_workers_sequential():
    train_image_classifier_sequential(worker_count=4)

    model = load_model(TEMP_DIR / "worker1/output", create_model)
    model_accuracy = evaluate_model(model, ExampleDirs.IMAGE_CLASSIFIER / "data")
    assert model_accuracy > 0.94


@with_temp_dir(clear_tmp_dir_end=False)
def test_one_worker_llm():
    example_dir = ExampleDirs.FINE_TUNE_LLM
    worker_count = 1

    p2p_network_simulator = multiprocessing.Process(
        target=simulate_p2p_network,
        kwargs=dict(
            example_dir=example_dir,
            worker_count=worker_count
        )
    )

    worker1 = multiprocessing.Process(
        target=run_node,
        kwargs=dict(
            role="WORKER-LLM",
            worker_count=worker_count,
            dir_name="worker1",
        )
    )

    worker1.start()
    p2p_network_simulator.start()
    worker1.join()
    p2p_network_simulator.join()
