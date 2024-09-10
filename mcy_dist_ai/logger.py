import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

stdout_handler = logging.StreamHandler()
stdout_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)
