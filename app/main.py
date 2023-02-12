import os
import json
from app.constants import IAS_API_KEY


def get_signature() -> str:
    with open("ias.sig", "r") as f:
        return f.read()


def get_certificate() -> str:
    with open("ias.cert", "r") as f:
        return f.read()


def get_report() -> dict:
    with open("ias.report", "r") as f:
        return json.loads(f.read())


def main():
    os.system("make distclean")
    os.system("make")

    os.system("gramine-sgx ./ra ra.py")
    os.system(f"gramine-sgx-ias-request report -k {IAS_API_KEY} -q gr.quote -r ias.report -s ias.sig -c ias.cert -v")

    return {
        "X-IASReport-Signature": get_signature(),
        "X-IASReport-Signing-Certificate": get_certificate(),
        "Body": get_report()
    }


if __name__ == "__main__":
    response = main()
    from pprint import pprint
    pprint(response)
