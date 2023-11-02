import os
import pickle
import time
from base64 import b64encode
from typing import Tuple, Any
from ecdsa.keys import SigningKey, VerifyingKey
from ecdsa.curves import NIST256p
from ecdsa.ecdh import ECDH
from cryptography.fernet import Fernet


def write(path: str, data: Any):
    directory_path = os.path.dirname(path)
    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(path, "w") as f:
        f.write(data)


def read(path: str) -> Any:
    with open(path, "r") as f:
        return f.read()


def wait_file(path: str):
    print(f"waiting for {path}")
    while not os.path.exists(path):
        time.sleep(1)
    print(f"{path} received")


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


def load_trained_model(path: str, shared_secret: str):
    with open(path, "rb") as file:
        encrypted_trained_model = pickle.load(file)
        trained_model = pickle.loads(decrypt(shared_secret, encrypted_trained_model))

    return trained_model
