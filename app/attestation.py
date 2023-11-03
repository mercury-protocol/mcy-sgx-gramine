import os
import time

from env import IAS_API_KEY


GR_QUOTE = "output/gr.quote"
IAS_REPORT = "output/ias.report"
IAS_SIGNATURE = "output/ias.sig"
IAS_CERTIFICATE = "output/ias.cert"


# TODO: attestation returns GROUP_OUT_OF_DATE
def remote_attestation():
    print(f"waiting for {GR_QUOTE}")
    while not os.path.exists(GR_QUOTE):
        time.sleep(1)
    print(f"{GR_QUOTE} received")

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
