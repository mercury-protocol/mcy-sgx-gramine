import os

from constants import LOCAL_PUBLIC_KEY_PATH
from utils import read


def startup():
    os.system("make distclean")
    os.system("make")
    os.system("gramine-sgx ./sgxapp startup.py")


def get_local_public_key():
    return read(LOCAL_PUBLIC_KEY_PATH)


def remote_attestation():
    os.system("python3 remote_attestation.py")


def verify_attestation():
    os.system("python3 verify_attestation.py")


def train_model():
    os.system("gramine-sgx ./sgxapp train_model.py")


if __name__ == "__main__":
    # TODO: these commands shall be performed at docker container startup
    os.system("/restart_aesm.sh")
    os.system("gramine-sgx-gen-private-key")

    startup()
    local_public_key = get_local_public_key()
    remote_attestation()  # TODO: attestation returns GROUP_OUT_OF_DATE
    verify_attestation()  # This has to be done on trusted machine other than the operator
    # train_model()

    # TODO: fix this issue related to train_model() run in container:
    # (NOTE: on local, the default sgx.max_threads = 4 is working,
    # but in container we have to set it to 32 and it's still considerably slower than in local)

    # error: There are no available TCS pages left for a new thread!
    # Please try to increase sgx.max_threads in the manifest.
    # The current value is 4
    # Segmentation fault (core dumped)
