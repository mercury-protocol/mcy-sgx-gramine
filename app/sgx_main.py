import os
from time import sleep

from sgx_constants import IAS_REPORT
from sgx_startup import startup
from sgx_train_model import train_model


startup()

while not os.path.exists(IAS_REPORT):
    sleep(1)

train_model()
