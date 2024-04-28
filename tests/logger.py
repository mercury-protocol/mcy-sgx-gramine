import colorlog
import logging

testlogger = logging.getLogger(__name__)
testlogger.setLevel(logging.INFO)

formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - TEST - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)

stdout_handler = logging.StreamHandler()
stdout_handler.setFormatter(formatter)

testlogger.addHandler(stdout_handler)
