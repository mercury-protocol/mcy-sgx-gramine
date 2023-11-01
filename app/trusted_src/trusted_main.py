import os
from time import sleep

from constants import ATTESTATION_REPORT_PATH
from startup import startup
from train_model import train_model


startup()

while not os.path.exists(ATTESTATION_REPORT_PATH):
    sleep(1)

train_model()
