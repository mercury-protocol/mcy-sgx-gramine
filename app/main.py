import os


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
    else:
        os.system("make distclean")

    os.system("make")
    os.system("gramine-sgx ./sgxapp")

# TODO: now we have to run remote attestation manually - do it in a separate thread once gr.quote is ready
