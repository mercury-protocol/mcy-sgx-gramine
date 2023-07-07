# add to sys.path only if using virtual env
import sys
# sys.path.append("/home/mercury/Documents/repos/mcy-sgx-gramine/venv/lib/python3.8/site-packages")


from constants import QUOTE_PATH, REPORT_PATH,  GR_QUOTE, LOCAL_SECRET
from utils import read, write, generate_key_pair, derive_public_from_secret


def save_local_secret():
    secret, public = generate_key_pair()
    write(LOCAL_SECRET, secret)


def save_quote():
    secret = read(LOCAL_SECRET)
    public = derive_public_from_secret(secret)
    write(REPORT_PATH, public)
    quote = read(QUOTE_PATH, binary=True)
    write(GR_QUOTE, quote, binary=True)


if __name__ == "__main__":
    save_local_secret()
    save_quote()


# ---------- DEBUG CODE ----------
# import os
# print("/dev/attestation: ", os.listdir("/dev/attestation"))
# print("/dev/attestation/keys: ", os.listdir("/dev/attestation/keys"))
# print("/dev/attestation/user_report_data: ", os.listdir("/dev/attestation/user_report_data"))
# print(read(KEY_PATH))
