import os

from constants import IAS_SIGNATURE, IAS_CERTIFICATE, IAS_REPORT, LOCAL_PUBLIC_KEY_PATH
from utils import read


def startup():
    os.system("make distclean")
    os.system("make")
    os.system("gramine-sgx ./sgxapp startup.py")


def get_local_public_key():
    return read(LOCAL_PUBLIC_KEY_PATH)


def remote_attestation():
    # TODO: attestation returns GROUP_OUT_OF_DATE - try to build without insecure configuration
    os.system("python3 remote_attestation.py")

    return {
        "X-IASReport-Signature": read(IAS_SIGNATURE).encode("utf-8").hex(),
        "X-IASReport-Signing-Certificate": read(IAS_CERTIFICATE).encode("utf-8").hex(),
        "Body": read(IAS_REPORT).encode("utf-8").hex()
    }


def train_model():
    os.system("gramine-sgx ./sgxapp train_model.py")


if __name__ == "__main__":
    startup()
    local_public_key = get_local_public_key()
    attestation_report = remote_attestation()
    train_model()

    # TODO: terminal command for destroy sgx
