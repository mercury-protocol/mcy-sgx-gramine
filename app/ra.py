# add to sys.path only if using virtual env
# import sys
# sys.path.append("/home/mercury/Documents/repos/mcy-sgx-gramine/venv/lib/python3.8/site-packages")

from ecdsa.curves import SECP256k1
from ecdsa import SigningKey
from hashlib import sha256

from constants import QUOTE_PATH, GR_QUOTE, KEY_PATH


def generate_key_pair():
    # TODO: only a 16 bytes long binary key can be saved
    # TODO: I could only add key file in manifest with an insecure option

    key = SigningKey.generate(curve=SECP256k1, hashfunc=sha256)

    # secret = key.to_string()
    # public = key.get_verifying_key().to_string()

    secret = bytes("0123456789abcdef", "utf-8")
    public = bytes("fedcba9876543210", "utf-8")

    return secret, public


def write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def read(path):
    with open(path, "rb") as f:
        return f.read()


def save_secret_key():
    sk, pk = generate_key_pair()
    write(KEY_PATH, sk)


def update_user_report_data():
    # TODO: finish implementation once key can be saved
    target_info = read("/dev/attestation/my_target_info")
    write("/dev/attestation/target_info", target_info)
    user_report_data = bytes("This is a Mercury operator.", "utf-8")
    write("/dev/attestation/user_report_data", user_report_data)


def get_quote():
    quote = read(QUOTE_PATH)
    write(GR_QUOTE, quote)


# TODO: attestation returns GROUP_OUT_OF_DATE - try to build without insecure configuration
save_secret_key()
update_user_report_data()
get_quote()


# ---------- DEBUG CODE ----------
# import os
# print("/dev/attestation: ", os.listdir("/dev/attestation"))
# print("/dev/attestation/keys: ", os.listdir("/dev/attestation/keys"))
# print("/dev/attestation/user_report_data: ", os.listdir("/dev/attestation/user_report_data"))
# print(read(KEY_PATH))

# if __name__ == "__main__":
#     sk, pk = generate_key_pair()
#     print(sk)
#     print(len(sk))
