# add to sys.path only if using virtual env
# import sys
# sys.path.append("/home/mercury/Documents/repos/mcy-sgx-gramine/venv/lib/python3.8/site-packages")


from constants import QUOTE_PATH, GR_QUOTE
from utils import read, write, generate_key_pair, derive_public_from_secret


def script1():
    # TODO: only a 16 bytes long binary key can be saved to /dev/attestation/keys/<key>-> ECDH P-256 generates 32 bytes
    # TODO: I could only add key file to that path in manifest with an insecure option
    # TODO: now we are saving the key to 'local_secret_key'

    secret, public = generate_key_pair()
    write("local_secret_key", secret)


def script2() -> bytes:
    secret = read("local_secret_key")
    public = derive_public_from_secret(secret)
    write("/dev/attestation/user_report_data", public)
    quote = read(QUOTE_PATH)
    return quote


def save_quote(quote: bytes):
    write(GR_QUOTE, quote)


script1()
quote = script2()
save_quote(quote)


# ---------- DEBUG CODE ----------
# import os
# print("/dev/attestation: ", os.listdir("/dev/attestation"))
# print("/dev/attestation/keys: ", os.listdir("/dev/attestation/keys"))
# print("/dev/attestation/user_report_data: ", os.listdir("/dev/attestation/user_report_data"))
# print(read(KEY_PATH))
