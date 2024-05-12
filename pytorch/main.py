import asyncio

from pytorch.constants import ROLE, LEADER_ROLE, WORKER_ROLE, WORKER_LLM_ROLE
from pytorch.exceptions import InvalidRole
from pytorch.worker import Worker
from pytorch.leader import Leader


def main():
    if ROLE == LEADER_ROLE:
        node = Leader()
    elif ROLE in (WORKER_ROLE, WORKER_LLM_ROLE):
        node = Worker()
    else:
        raise InvalidRole

    asyncio.run(node.run())


if __name__ == "__main__":
    main()
