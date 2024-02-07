import asyncio

from pytorch.constants import ROLE, WORKER_ROLE, LEADER_ROLE
from pytorch.worker import Worker
from pytorch.leader import Leader


def main():
    if ROLE == LEADER_ROLE:
        node = Leader()
    elif ROLE == WORKER_ROLE:
        node = Worker()
    else:
        raise Exception("Invalid role.")  # TODO: implement exceptions

    print(ROLE)  # TODO: implement proper logger
    asyncio.run(node.run())


if __name__ == "__main__":
    main()
