import os

from utils import read
from constants import IAS_API_KEY, GR_QUOTE, IAS_REPORT, IAS_SIGNATURE, IAS_CERTIFICATE


def remote_attestation():
    # TODO: attestation returns GROUP_OUT_OF_DATE - try to build without insecure configuration
    os.system(f"gramine-sgx-ias-request report"
              f" -k {IAS_API_KEY}"
              f" -q {GR_QUOTE}"
              f" -r {IAS_REPORT}"
              f" -s {IAS_SIGNATURE}"
              f" -c {IAS_CERTIFICATE} -v")

    return {
        "X-IASReport-Signature": read(IAS_SIGNATURE).encode("utf-8").hex(),
        "X-IASReport-Signing-Certificate": read(IAS_CERTIFICATE).encode("utf-8").hex(),
        "Body": read(IAS_REPORT).encode("utf-8").hex()
    }
