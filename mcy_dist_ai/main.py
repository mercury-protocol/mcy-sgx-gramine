import asyncio

from mcy_dist_ai.args import ROLE
from mcy_dist_ai.constants import LEADER_ROLE, WORKER_ROLE
from mcy_dist_ai.exceptions import InvalidRole
from mcy_dist_ai.worker import Worker
from mcy_dist_ai.leader import Leader


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
