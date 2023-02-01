import os
from gramine_examples.secrets import *  # not included in git


MINIMAL = "secret_prov_minimal"
NORMAL = "secret_prov"
FULL = "secret_prov_pf"

os.chdir("ra-tls-secret-prov")


def perform_commands(commands: list):
    cmd_color = '\033[96m'
    color_end = '\033[0m'
    for cmd in commands:
        print()
        print(f"{cmd_color}{cmd}{color_end}")
        os.system(cmd)


def epid(level: str = MINIMAL, linkable: bool = False):
    spid = SPID_L if linkable else SPID_U
    ias_api_key = IAS_PRIMARY_KEY_L if linkable else IAS_PRIMARY_KEY_U
    commands_build = [
        "make clean",
        f"make app epid RA_TYPE=epid RA_CLIENT_SPID={spid} RA_CLIENT_LINKABLE={int(linkable)}"
    ]
    commands_run = [
        "RA_TLS_ALLOW_DEBUG_ENCLAVE_INSECURE=1",  # WARNING: Don't use this option in production!
        "RA_TLS_ALLOW_OUTDATED_TCB_INSECURE=1",  # WARNING: Don't use this option in production!
        f"A_TLS_EPID_API_KEY={ias_api_key}",
        "./server_epid wrap_key &"
        "gramine-sgx ./client",
        "kill %%"
    ]

    perform_commands(commands_build)
    os.chdir(level)
    perform_commands(commands_run)


def dcap(level: str = MINIMAL):
    commands_build = [
        "make clean",
        "make app dcap RA_TYPE=dcap"
    ]
    commands_run = [
        "RA_TLS_ALLOW_DEBUG_ENCLAVE_INSECURE=1",  # WARNING: Don't use this option in production!
        "RA_TLS_ALLOW_OUTDATED_TCB_INSECURE=1",  # WARNING: Don't use this option in production!
        "./server_dcap wrap_key &",
        "gramine-sgx ./client",
        "kill %%"
    ]

    perform_commands(commands_build)
    os.chdir(level)
    perform_commands(commands_run)


if __name__ == "__main__":
    """Choose which type of RA to use by commenting the other."""
    # epid(level=FULL, linkable=False)
    dcap(level=FULL)
