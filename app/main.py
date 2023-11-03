import os
import threading
from attestation import remote_attestation
from env import RA_CLIENT_SPID


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


def run_sgxapp():
    os.system("gramine-sgx ./sgxapp")


if __name__ == "__main__":
    if is_running_in_docker():
        os.system("/restart_aesm.sh")
        os.system("gramine-sgx-gen-private-key")
    else:
        os.system("make distclean")

    os.system(f"make RA_CLIENT_SPID={RA_CLIENT_SPID}")

    sgxapp_thread = threading.Thread(target=run_sgxapp)
    remote_attestation_thread = threading.Thread(target=remote_attestation)

    sgxapp_thread.start()
    remote_attestation_thread.start()

    sgxapp_thread.join()
    remote_attestation_thread.join()
