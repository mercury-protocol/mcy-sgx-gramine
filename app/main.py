import os
import json
from app.constants import IAS_API_KEY, GR_QUOTE, IAS_REPORT, IAS_SIGNATURE, IAS_CERTIFICATE


def get_signature() -> str:
    with open(IAS_SIGNATURE, "r") as f:
        return f.read()


def get_certificate() -> str:
    with open(IAS_CERTIFICATE, "r") as f:
        return f.read()


def get_report() -> dict:
    with open(IAS_REPORT, "r") as f:
        return json.loads(f.read())


def main():
    os.system("make distclean")
    os.system("make")

    os.system("gramine-sgx ./ra ra.py")
    # TODO: attestation returns GROUP_OUT_OF_DATE - try to build without insecure configuration
    os.system(f"gramine-sgx-ias-request report"
              f" -k {IAS_API_KEY}"
              f" -q {GR_QUOTE}"
              f" -r {IAS_REPORT}"
              f" -s {IAS_SIGNATURE}"
              f" -c {IAS_CERTIFICATE} -v")

    return {
        "X-IASReport-Signature": get_signature(),
        "X-IASReport-Signing-Certificate": get_certificate(),
        "Body": get_report()
    }


if __name__ == "__main__":
    response = main()
    from pprint import pprint
    pprint(response)
