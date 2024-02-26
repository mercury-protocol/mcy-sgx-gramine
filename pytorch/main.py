import asyncio

from pytorch.constants import ROLE, LEADER_ROLE, WORKER_ROLE
from pytorch.exceptions import InvalidRole
from pytorch.worker import Worker
from pytorch.leader import Leader

# TODO: fix unused import
# this has to be imported even if we don't use it here, otherwise python throws error
from mcy_dist_ai.data_partitioner import Partition, DataPartitioner


def main():
    if ROLE == LEADER_ROLE:
        node = Leader()
    elif ROLE == WORKER_ROLE:
        node = Worker()
    else:
        raise InvalidRole

    asyncio.run(node.run())


if __name__ == "__main__":
    main()
