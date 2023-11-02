import os
import json

from utils import read, write_json
from constants import IAS_API_KEY, GR_QUOTE, IAS_REPORT, IAS_SIGNATURE, IAS_CERTIFICATE, ATTESTATION_REPORT_PATH


# TODO: attestation returns GROUP_OUT_OF_DATE
def remote_attestation():
    os.system("gramine-sgx-ias-request report"
              f" -k {IAS_API_KEY}"
              f" -q {GR_QUOTE}"
              f" -r {IAS_REPORT}"
              f" -s {IAS_SIGNATURE}"
              f" -c {IAS_CERTIFICATE}")

    # TODO: maybe the user app could just read the 3 files and this write operation could be spared
    attestation_report = {
        "X-IASReport-Signature": read(IAS_SIGNATURE),
        "X-IASReport-Signing-Certificate": read(IAS_CERTIFICATE),
        "Body": json.loads(read(IAS_REPORT))
    }
    write_json(ATTESTATION_REPORT_PATH, attestation_report)


if __name__ == "__main__":
    remote_attestation()
