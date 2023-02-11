# from ecdsa.curves import SECP256k1
# from ecdsa import SigningKey
# from hashlib import sha256
# import sys
# sys.path.append("/home/mercury/Documents/repos/mcy-sgx-gramine/venv/lib/python3.8/site-packages")
import os

REPORT_PATH = "/dev/attestation/user_report_data"
QUOTE_PATH = "/dev/attestation/quote"
KEY_NAME = "ecdsa_secret"
KEY_PATH = f"/dev/attestation/keys/{KEY_NAME}"


def generate_key_pair():
    # key = SigningKey.generate(curve=SECP256k1, hashfunc=sha256)
    # public = key.get_verifying_key().to_string()
    # secret = key.to_string()
    # return secret, public
    return bytes("secret", "utf-8"), bytes("public", "utf-8")


def write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def read(path):
    with open(path, "r") as f:
        return f.read()


def script1():
    sk, pk = generate_key_pair()
    print(os.listdir("/dev/attestation"))
    print(os.listdir("/dev/attestation/keys"))
    print(read("/dev/attestation/quote"))  # PermissionError: [Errno 13] Permission denied: '/dev/attestation/quote'
                                           # we cannot run python script as root
    print(read("/dev/attestation/attestation_type"))

    # write_file(KEY_PATH, sk)
    # write_file("key.txt", sk)


# write_report()
# print(get_quote())

script1()


"""
make
gramine-sgx ./main main.py
"""
