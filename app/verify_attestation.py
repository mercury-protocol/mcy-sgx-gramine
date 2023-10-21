import os

from constants import IAS_REPORT, IAS_SIGNATURE


def verify_attestation():
    os.system("gramine-sgx-ias-verify-report "
              f"-r {IAS_REPORT} "
              f"-s {IAS_SIGNATURE}")


if __name__ == "__main__":
    verify_attestation()
