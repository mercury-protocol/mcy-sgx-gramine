import os

from trusted_src.constants import LOCAL_PUBLIC_KEY_PATH


def startup():
    os.system("make distclean")
    os.system("make")
    os.system("gramine-sgx ./sgxapp trusted_src/startup.py")


def get_local_public_key():
    with open(LOCAL_PUBLIC_KEY_PATH, "r") as file:
        return file.read()


def remote_attestation():
    os.system("python3 trusted_src/remote_attestation.py")


def verify_attestation():
    os.system("python3 trusted_src/verify_attestation.py")


def train_model():
    os.system("gramine-sgx ./sgxapp trusted_src/train_model.py")


def is_running_in_docker():
    try:
        with open('/proc/1/cgroup', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'docker' in line:
                    return True
    except FileNotFoundError:
        pass
    return False


if __name__ == "__main__":
    if is_running_in_docker():
        os.system("/restart_aesm.sh")
        os.system("gramine-sgx-gen-private-key")

    # os.system("make")
    # os.system("gramine-sgx ./sgxapp")
    startup()
    local_public_key = get_local_public_key()
    remote_attestation()  # TODO: attestation returns GROUP_OUT_OF_DATE
    verify_attestation()  # This has to be done on trusted machine other than the operator
    train_model()
