import sys
from argparse import ArgumentParser

from mcy_dist_ai.constants import LEADER_ROLE, WORKER_ROLE
from mcy_dist_ai.logger import logger


parser = ArgumentParser()
parser.add_argument("--role", type=str, help="Node role - leader or worker")
parser.add_argument("--worker_count", type=int, help="Worker nodes count")
parser.add_argument(
    "--tensor_load",
    action='store_true',
    default=False,
    help="pass this arg when data was split by mcy script"
)
args = parser.parse_args()
if args.role is None:
    logger.error("Role argument is missing")
    sys.exit(1)
if args.role.upper() not in (LEADER_ROLE, WORKER_ROLE):
    logger.error(f"role must be {LEADER_ROLE} or {WORKER_ROLE}")
    sys.exit(1)
if args.role == LEADER_ROLE and args.worker_count is None:
    logger.error("Worker nodes count argument is required for leader")
    sys.exit(1)

ROLE = args.role.upper()
WORKER_NODES_NUM = int(args.worker_count)
if ROLE == LEADER_ROLE and WORKER_NODES_NUM == 1:
    logger.info("Leader is not starting because there's only one worker.")
    sys.exit(0)
TENSOR_LOAD = args.tensor_load
