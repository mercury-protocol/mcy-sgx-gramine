import os

from env import IAS_API_KEY
from sgx_constants import GR_QUOTE, IAS_REPORT, IAS_SIGNATURE, IAS_CERTIFICATE
from sgx_utils import wait_file


# TODO: attestation returns GROUP_OUT_OF_DATE
def remote_attestation():
    wait_file(GR_QUOTE)
    os.system("gramine-sgx-ias-request report"
              f" -k {IAS_API_KEY}"
              f" -q {GR_QUOTE}"
              f" -r {IAS_REPORT}"
              f" -s {IAS_SIGNATURE}"
              f" -c {IAS_CERTIFICATE}")


def verify_attestation():
    os.system("gramine-sgx-ias-verify-report "
              f"-r {IAS_REPORT} "
              f"-s {IAS_SIGNATURE}")


if __name__ == "__main__":
    verify_attestation()
