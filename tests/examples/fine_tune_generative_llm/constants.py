from bitsandbytes.optim import PagedAdamW
from transformers import AdamW

from tests.constants import ExampleDirs, TRAINED_MODEL_FILE


FINE_TUNE_GENERATIVE_LLM_DIR = ExampleDirs.FINE_TUNE_GENERATIVE_LLM
SPLIT_DATA_PATH = FINE_TUNE_GENERATIVE_LLM_DIR / "split_data"
DATA_PATH = FINE_TUNE_GENERATIVE_LLM_DIR / "data"
RAW_DATA_PATH = FINE_TUNE_GENERATIVE_LLM_DIR / "raw_data"
TRAINED_MODEL_PATH = FINE_TUNE_GENERATIVE_LLM_DIR / TRAINED_MODEL_FILE

VALID_DATA_SPLIT_PARTITIONS = (1, 2, 4, 5, 8)


DATA = "mlabonne/guanaco-llama2-1k"
MODEL = "NousResearch/Llama-2-7b-chat-hf"
# MODEL = "google-bert/bert-base-cased"
OPTIMIZER = "adamw"  # "paged_adamw_32bit"
OPTIMIZER_MAPPING = {
    "adamw": AdamW,
    "paged_adamw_32bit": PagedAdamW,
}
