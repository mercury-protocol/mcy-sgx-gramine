import os

from constants import IAS_API_KEY, GR_QUOTE, IAS_REPORT, IAS_SIGNATURE, IAS_CERTIFICATE


# TODO: attestation returns GROUP_OUT_OF_DATE
def remote_attestation():
    os.system("gramine-sgx-ias-request report"
              f" -k {IAS_API_KEY}"
              f" -q {GR_QUOTE}"
              f" -r {IAS_REPORT}"
              f" -s {IAS_SIGNATURE}"
              f" -c {IAS_CERTIFICATE}")


if __name__ == "__main__":
    remote_attestation()
