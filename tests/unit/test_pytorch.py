from unittest.mock import patch

from tests.utils import pytorch_context


@pytorch_context(role="WORKER", worker_count=1, user_script="image_classifier.py")
def test_one_worker():
    import pytorch.utils
    print("END")
