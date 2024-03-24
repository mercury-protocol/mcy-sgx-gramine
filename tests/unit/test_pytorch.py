from tests.constants import EXAMPLES_DIR, TEMP_DIR
from tests.examples.image_classifier.user_script import create_model
from tests.utils import run_node, load_model, evaluate_model


def test_one_worker_image_classifier():
    import multiprocessing

    worker1 = multiprocessing.Process(
        target=run_node,
        kwargs=dict(
            role="WORKER",
            worker_count=1,
            temp_dir_name="worker1",
            example_dir="image_classifier",
            clear_tmp_dir_end=False
        )
    )

    worker2 = multiprocessing.Process(
        target=run_node,
        kwargs=dict(
            role="WORKER",
            worker_count=1,
            temp_dir_name="worker2",
            example_dir="image_classifier",
            clear_tmp_dir_end=False
        )
    )

    worker1.start()
    worker2.start()
    worker1.join()
    worker2.join()

    model = load_model(TEMP_DIR / "worker1/output", create_model)
    evaluate_model(model, EXAMPLES_DIR / "image_classifier/data")
    model = load_model(TEMP_DIR / "worker2/output", create_model)
    evaluate_model(model, EXAMPLES_DIR / "image_classifier/data")


def test_one_worker_llm():
    run_node(
        role="WORKER-LLM",
        worker_count=1,
        temp_dir_name="worker1",
        example_dir="fine_tune_llm",
        clear_tmp_dir_end=False
    )
