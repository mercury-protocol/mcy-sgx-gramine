from tests.utils import pytorch_context


@pytorch_context(
    role="WORKER", worker_count=1,
    example_dir="image_classifier",
    clear_tmp_dir_end=False
)
def test_one_worker():
    from pytorch.main import main
    main()


@pytorch_context(
    role="WORKER-LLM", worker_count=1,
    example_dir="fine_tune_llm",
    clear_tmp_dir_end=False
)
def test_one_worker_llm():
    from pytorch.main import main
    main()
