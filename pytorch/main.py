import asyncio

from pytorch.constants import AGGREGATED_STATE_DICT_PATH
from pytorch.helpers.eval import make_predictions
from pytorch.helpers.test_network import test, test_data_loader
from pytorch.utils import load_network, list_worker_nodes
from pytorch.worker import Worker
from pytorch.leader import Leader
from remote.utils import MeasureTime


MAKE_PREDICTIONS = False


async def train_network():
    leader_task = asyncio.create_task(Leader().aggregate_network())
    worker_tasks = [
        asyncio.create_task(Worker(node).train_network())
        for node in list_worker_nodes()
    ]
    await asyncio.gather(leader_task, *worker_tasks)


def evaluate_network(state_dict_path=AGGREGATED_STATE_DICT_PATH):
    network = load_network(state_dict_path)
    test(network)
    if MAKE_PREDICTIONS:
        make_predictions(network, test_data_loader)


if __name__ == "__main__":
    with MeasureTime("train_network()"):
        asyncio.run(train_network())
    evaluate_network()
