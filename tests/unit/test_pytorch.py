from tests.conftest import pytorch_context
from tests.constants import EXAMPLES_DIR
from tests.utils import load_model, evaluate_model


@pytorch_context(
    role="WORKER", worker_count=1,
    example_dir="image_classifier",
    clear_tmp_dir_end=False
)
def test_one_worker_image_classifier():
    from pytorch.main import main
    from tests.examples.image_classifier.user_script import create_model

    main()

    model = load_model(create_model)
    evaluate_model(model, EXAMPLES_DIR + "/image_classifier/data")


@pytorch_context(
    role="WORKER-LLM", worker_count=1,
    example_dir="fine_tune_llm",
    clear_tmp_dir_end=False
)
def test_one_worker_llm():
    from pytorch.main import main
    main()
