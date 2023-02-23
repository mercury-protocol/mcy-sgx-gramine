from ecdsa.keys import SigningKey, VerifyingKey
from ecdsa.curves import NIST256p
from ecdsa.ecdh import ECDH
from typing import Tuple, Any
from constants import LOCAL_SECRET


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


def generate_shared_secret(remote_public_key: str):
    secret = read(LOCAL_SECRET)
    public = derive_public_from_secret(secret)

    secret = SigningKey.from_string(bytearray.fromhex(secret), curve=NIST256p)
    public = VerifyingKey.from_string(bytearray.fromhex(public), curve=NIST256p)

    ecdh = ECDH(curve=NIST256p, public_key=public, private_key=secret)
    ecdh.load_received_public_key_bytes(bytearray.fromhex(remote_public_key))

    shared_secret = ecdh.generate_sharedsecret_bytes().hex()

    return shared_secret
