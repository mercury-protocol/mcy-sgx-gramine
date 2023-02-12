import sys
import os

sys.path.append("/home/mercury/Documents/repos/mcy-sgx-gramine/venv/lib/python3.8/site-packages")

from ecdsa.curves import SECP256k1
from ecdsa import SigningKey
from hashlib import sha256


REPORT_PATH = "/dev/attestation/user_report_data"
QUOTE_PATH = "/dev/attestation/quote"
KEY_NAME = "ecdsa_secret"
KEY_PATH = f"/dev/attestation/keys/{KEY_NAME}"


def generate_key_pair():
    key = SigningKey.generate(curve=SECP256k1, hashfunc=sha256)
    public = key.get_verifying_key().to_string()
    secret = key.to_string()
    return secret, public
    # return bytes("1"*16, "utf-8"), bytes("public", "utf-8")


def write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def read(path):
    with open(path, "rb") as f:
        return f.read()


def script1():
    sk, pk = generate_key_pair()

    print("/: ", os.listdir("/"))
    print(len(sk))
    # print("/dev/attestation: ", os.listdir("/dev/attestation"))
    # print("/dev/attestation/keys: ", os.listdir("/dev/attestation/keys"))
    # print("/dev/attestation/quote: ", read("/dev/attestation/quote"))

    # write(KEY_PATH, sk)  # PermissionError: [Errno 13] Permission denied: 'key.txt'
    write("key.txt", sk)  #PermissionError: [Errno 13] Permission denied: 'key.txt'
    print(read("key.txt"))


# write_report()
# print(get_quote())

script1()


"""
make
gramine-sgx ./main main.py
"""
