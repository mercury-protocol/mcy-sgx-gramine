from tests.constants import TEMP_OUTPUT_DIR, EXAMPLES_DIR
from tests.tools import evaluate_model
from tests.utils import pytorch_context


@pytorch_context(
    role="WORKER", worker_count=1,
    example_dir="image_classifier",
    clear_tmp_dir_end=False
)
def test_one_worker_image_classifier():
    from pytorch.main import main
    from tests.examples.image_classifier.user_script import create_model
    import torch
    main()
    model = create_model()
    model.load_state_dict(torch.load(TEMP_OUTPUT_DIR + "/trained_model.pth"))
    evaluate_model(model, EXAMPLES_DIR + "/image_classifier/data")


@pytorch_context(
    role="WORKER-LLM", worker_count=1,
    example_dir="fine_tune_llm",
    clear_tmp_dir_end=False
)
def test_one_worker_llm():
    from pytorch.main import main
    main()
