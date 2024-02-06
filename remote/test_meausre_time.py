from io import StringIO

from model import run
from utils import read, MeasureTime
from constants import DATA_PATH, MODEL_PATH


def train_model():
    with MeasureTime("train_model()"):
        run(DATA_PATH)


def train_model_eval():
    data = read(DATA_PATH)
    model = read(MODEL_PATH)

    with MeasureTime("train_model_eval()"):
        exec(model)
        data = StringIO(data)
        eval("run(data)")


if __name__ == "__main__":
    train_model()
    train_model_eval()

