import asyncio

import pytorch.constants as constants 
from pytorch.exceptions import InvalidRole
from pytorch.worker import Worker
from pytorch.leader import Leader
from mcy_dist_ai.data_partitioner import Partition, DataPartitioner # this has to be imported even if we don't use it here. otherwise python throws error
    
def main():
    if constants.ROLE == constants.LEADER_ROLE:
        node = Leader()
    elif constants.ROLE == constants.WORKER_ROLE:
        node = Worker()
    else:
        raise InvalidRole

    asyncio.run(node.run())


if __name__ == "__main__":
    main()
