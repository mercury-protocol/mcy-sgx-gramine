import os
from time import sleep

from constants import IAS_REPORT
from startup import startup
from train_model import train_model


startup()

while not os.path.exists(IAS_REPORT):
    sleep(1)

train_model()
