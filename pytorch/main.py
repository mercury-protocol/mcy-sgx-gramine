import asyncio

from pytorch.constants import AGGREGATED_STATE_DICT_PATH
from pytorch.helpers.eval import make_predictions
from pytorch.helpers.test_network import test, test_data_loader
from pytorch.utils import load_network, list_worker_nodes
from pytorch.worker import Worker
from pytorch.leader import Leader


MAKE_PREDICTIONS = False


async def full_train():
    leader = asyncio.create_task(Leader().aggregate_network())
    workers = list()
    for node in list_worker_nodes():
        workers.append(asyncio.create_task(Worker(node).train_network()))

    await asyncio.gather(leader, *workers)


def evaluate_network(state_dict_path=AGGREGATED_STATE_DICT_PATH):
    network = load_network(state_dict_path)
    test(network)
    if MAKE_PREDICTIONS:
        make_predictions(network, test_data_loader)


if __name__ == "__main__":
    asyncio.run(full_train())
    evaluate_network()
