from ecdsa.keys import SigningKey
from ecdsa.curves import NIST256p
from ecdsa.ecdh import ECDH
from typing import Tuple


def write(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def generate_key_pair() -> Tuple[bytes, bytes]:
    ecdh = ECDH(curve=NIST256p)
    ecdh.generate_private_key()
    secret = ecdh.private_key.to_string()
    public = ecdh.get_public_key().to_string()

    return secret, public


def derive_public_from_secret(secret: bytes) -> bytes:
    secret = SigningKey.from_string(secret, curve=NIST256p)
    return secret.get_verifying_key().to_string()
