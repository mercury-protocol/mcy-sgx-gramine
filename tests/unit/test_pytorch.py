import multiprocessing

from tests.constants import TEMP_DIR, ExampleDirs
from tests.examples.image_classifier.user_script import create_model
from tests.examples.image_classifier.data_manipulation.split_mnist_image_data import split_and_save_data
from tests.simulate_p2p_network import simulate_p2p_network
from tests.utils import run_node, load_model, evaluate_model, with_temp_dir


@with_temp_dir(clear_tmp_dir_end=False)
def train_image_classifier(worker_count: int):
    example_dir = ExampleDirs.IMAGE_CLASSIFIER
    workers = []

    if worker_count > 1:
        split_and_save_data(split_into=worker_count, random_seed=42)

    for i in range(worker_count):
        workers.append(
            multiprocessing.Process(
                name=f"worker{i+1}",
                target=run_node,
                kwargs=dict(
                    role="WORKER",
                    worker_count=worker_count,
                    dir_name=f"worker{i+1}",
                )
            )
        )

    p2p_network_simulator = multiprocessing.Process(
        name="p2p_network_simulator",
        target=simulate_p2p_network,
        kwargs=dict(
            example_dir=example_dir,
            worker_count=worker_count
        )
    )

    leader = multiprocessing.Process(
        name="leader",
        target=run_node,
        kwargs=dict(
            role="LEADER",
            worker_count=worker_count,
            dir_name="leader",
        )
    )

    [worker.start() for worker in workers]
    leader.start()
    p2p_network_simulator.start()
    [worker.join() for worker in workers]
    leader.join()
    p2p_network_simulator.join()


def test_train_image_classifier_one_worker():
    train_image_classifier(worker_count=1)

    model = load_model(TEMP_DIR / "worker1/output", create_model)
    model_accuracy = evaluate_model(model, ExampleDirs.IMAGE_CLASSIFIER / "data")
    assert model_accuracy > 0.94


def test_train_image_classifier_two_workers():
    train_image_classifier(worker_count=2)

    model = load_model(TEMP_DIR / "leader/output", create_model)
    model_accuracy = evaluate_model(model, ExampleDirs.IMAGE_CLASSIFIER / "data")
    assert model_accuracy > 0.94  # TODO: FAIL: accuracy reduced to 0.91


def test_train_image_classifier_four_workers():
    train_image_classifier(worker_count=4)

    model = load_model(TEMP_DIR / "leader/output", create_model)
    model_accuracy = evaluate_model(model, ExampleDirs.IMAGE_CLASSIFIER / "data")
    assert model_accuracy > 0.94  # TODO: FAIL: accuracy reduced to 0.84!!


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
