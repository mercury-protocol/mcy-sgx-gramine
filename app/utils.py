from ecdsa.keys import SigningKey
from ecdsa.curves import NIST256p
from ecdsa.ecdh import ECDH
from typing import Tuple, Any


def write(path: str, data: Any, binary: bool = False):
    mode = "wb" if binary else "w"
    with open(path, mode) as f:
        f.write(data)


def read(path: str, binary: bool = False) -> Any:
    mode = "rb" if binary else "r"
    with open(path, mode) as f:
        return f.read()


def generate_key_pair() -> Tuple[str, str]:
    ecdh = ECDH(curve=NIST256p)
    ecdh.generate_private_key()
    secret = ecdh.private_key.to_string().hex()
    public = ecdh.get_public_key().to_string().hex()

    return secret, public


def derive_public_from_secret(secret: str) -> str:
    secret = SigningKey.from_string(bytearray.fromhex(secret), curve=NIST256p)
    return secret.get_verifying_key().to_string().hex()
