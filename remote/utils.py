from base64 import b64encode
from typing import Tuple, Any

from ecdsa.keys import SigningKey, VerifyingKey
from ecdsa.curves import NIST256p
from ecdsa.ecdh import ECDH
from cryptography.fernet import Fernet


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


def derive_shared_secret(secret: str, public: str, remote_public: str) -> str:
    secret = SigningKey.from_string(bytearray.fromhex(secret), curve=NIST256p)
    public = VerifyingKey.from_string(bytearray.fromhex(public), curve=NIST256p)

    ecdh = ECDH(curve=NIST256p, public_key=public, private_key=secret)
    ecdh.load_received_public_key_bytes(bytearray.fromhex(remote_public))

    shared_secret = ecdh.generate_sharedsecret_bytes().hex()

    return shared_secret


def encrypt(shared_secret: str, decrypted: bytes) -> bytes:
    shared_secret = b64encode(bytearray.fromhex(shared_secret))
    fernet = Fernet(shared_secret)
    encrypted = fernet.encrypt(decrypted)
    return encrypted


def decrypt(shared_secret: str, encrypted: bytes) -> bytes:
    shared_secret = b64encode(bytearray.fromhex(shared_secret))
    fernet = Fernet(shared_secret)
    decrypted = fernet.decrypt(encrypted)
    return decrypted
