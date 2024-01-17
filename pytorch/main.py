import asyncio

from pytorch.constants import AGGREGATED_STATE_DICT_PATH
from pytorch.helpers.eval import evaluate_network, make_predictions, test_data_loader
from pytorch.helpers.simulate_p2p_network import simulate_p2p_network
from pytorch.utils import load_network, list_worker_nodes
from pytorch.worker import Worker
from pytorch.leader import Leader
from remote.utils import MeasureTime


async def main():
    p2p_network_task = asyncio.create_task(simulate_p2p_network())
    leader_task = asyncio.create_task(Leader().run())
    worker_tasks = [
        asyncio.create_task(Worker(node).run())
        for node in list_worker_nodes()
    ]
    await asyncio.gather(p2p_network_task, leader_task, *worker_tasks)


if __name__ == "__main__":
    with MeasureTime("train_network()"):
        asyncio.run(main())

    network = load_network(AGGREGATED_STATE_DICT_PATH)
    evaluate_network(network)
    # make_predictions(network, test_data_loader)
