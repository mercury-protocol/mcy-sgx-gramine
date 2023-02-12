import sys
sys.path.append("/home/mercury/Documents/repos/mcy-sgx-gramine/venv/lib/python3.8/site-packages")

from ecdsa.curves import SECP256k1
from ecdsa import SigningKey
from hashlib import sha256

from constants import QUOTE_PATH, KEY_PATH, GR_QUOTE


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


def get_quote():
    quote = read(QUOTE_PATH)
    write(GR_QUOTE, quote)


get_quote()
# sk, pk = generate_key_pair()

# print("/dev/attestation: ", os.listdir("/dev/attestation"))
# print("/dev/attestation/keys: ", os.listdir("/dev/attestation/keys"))
