import os
from gramine_examples.secrets import *  # not included in git

os.chdir("ra-tls-secret-prov")


def perform_commands(commands: list):
    for cmd in commands:
        os.system(cmd)


def epid(linkable: bool = False):
    spid = SPID_L if linkable else SPID_U
    primary_key = IAS_PRIMARY_KEY_L if linkable else IAS_PRIMARY_KEY_U
    secondary_key = IAS_SECONDARY_KEY_L if linkable else IAS_SECONDARY_KEY_U
    commands_build = [
        "make clean",
        f"make app epid RA_TYPE=epid RA_CLIENT_SPID={spid} RA_CLIENT_LINKABLE={int(linkable)}"
    ]
    commands_run = [
        "RA_TLS_ALLOW_DEBUG_ENCLAVE_INSECURE=1",  # WARNING: Don't use this option in production!
        "RA_TLS_ALLOW_OUTDATED_TCB_INSECURE=1",  # WARNING: Don't use this option in production!
        f"A_TLS_EPID_API_KEY={primary_key}",
        "./server_epid wrap_key &",
        "gramine-sgx ./client",
        "kill %%"
    ]

    perform_commands(commands_build)
    os.chdir("secret_prov_minimal")
    perform_commands(commands_run)


def dcap():
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
    os.chdir("secret_prov_minimal")
    perform_commands(commands_run)


if __name__ == "__main__":
    """Choose which type of RA to use by commenting the other."""
    epid(linkable=True)
    # dcap()
