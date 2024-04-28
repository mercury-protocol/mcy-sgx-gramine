import multiprocessing

from tests.constants import TEMP_DIR, ExampleDirs
from tests.examples.image_classifier.user_script import create_model
from tests.simulate_p2p_network import simulate_p2p_network
from tests.utils import run_node, load_model, evaluate_model, with_temp_dir


@with_temp_dir(clear_tmp_dir_end=False)
def test_one_worker_image_classifier():
    example_dir = ExampleDirs.IMAGE_CLASSIFIER
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
            role="WORKER",
            worker_count=worker_count,
            dir_name="worker1",
        )
    )

    worker1.start()
    p2p_network_simulator.start()
    worker1.join()
    p2p_network_simulator.join()

    model = load_model(TEMP_DIR / "worker1/output", create_model)
    worker1_accuracy = evaluate_model(model, example_dir / "data")
    assert worker1_accuracy > 0.94


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
