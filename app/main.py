import os


def startup():
    os.system("make distclean")
    os.system("make")
    os.system("gramine-sgx ./sgxapp startup.py")


def remote_attestation():
    # TODO: attestation returns GROUP_OUT_OF_DATE - try to build without insecure configuration
    return os.system("gramine-sgx ./sgxapp remote_attestation.py")


def train_model():
    os.system("gramine-sgx ./sgxapp train_model.py")


if __name__ == "__main__":
    startup()
    attestation_report = remote_attestation()
    train_model()

    # TODO: terminal command for destroy sgx
