# add to sys.path only if using virtual env
# import sys
# sys.path.append("/home/mercury/Documents/repos/mcy-sgx-gramine/venv/lib/python3.8/site-packages")


from constants import QUOTE_PATH, GR_QUOTE, SEALED_LOCAL_KEY
from utils import read, write, generate_key_pair, derive_public_from_secret


def script1():
    secret, public = generate_key_pair()
    write(SEALED_LOCAL_KEY, secret)


def script2() -> bytes:
    secret = read(SEALED_LOCAL_KEY)
    public = derive_public_from_secret(secret)
    write("/dev/attestation/user_report_data", public)
    quote = read(QUOTE_PATH, binary=True)
    return quote


def save_quote(quote: bytes):
    write(GR_QUOTE, quote, binary=True)


script1()
quote = script2()
save_quote(quote)


# ---------- DEBUG CODE ----------
# import os
# print("/dev/attestation: ", os.listdir("/dev/attestation"))
# print("/dev/attestation/keys: ", os.listdir("/dev/attestation/keys"))
# print("/dev/attestation/user_report_data: ", os.listdir("/dev/attestation/user_report_data"))
# print(read(KEY_PATH))
