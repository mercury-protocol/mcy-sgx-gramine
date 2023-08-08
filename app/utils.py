import json
from base64 import b64encode
from typing import Tuple, Any

from ecdsa.keys import SigningKey, VerifyingKey
from ecdsa.curves import NIST256p
from ecdsa.ecdh import ECDH
from cryptography.fernet import Fernet

from constants import LOCAL_SECRET_KEY_PATH, SIMULATED_RECEIVE


def write(path: str, data: Any, binary: bool = False):
    mode = "wb" if binary else "w"
    with open(path, mode) as f:
        f.write(data)


def read(path: str, binary: bool = False) -> Any:
    mode = "rb" if binary else "r"
    with open(path, mode) as f:
        return f.read()


def write_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def generate_key_pair() -> Tuple[str, str]:
    ecdh = ECDH(curve=NIST256p)
    ecdh.generate_private_key()
    secret = ecdh.private_key.to_string().hex()
    public = ecdh.get_public_key().to_string().hex()

    return secret, public


def derive_public_from_secret(secret: str) -> str:
    secret = SigningKey.from_string(bytearray.fromhex(secret), curve=NIST256p)
    return secret.get_verifying_key().to_string().hex()


def derive_shared_secret(remote_public: str):
    local_secret = read(LOCAL_SECRET_KEY_PATH)
    local_public = derive_public_from_secret(local_secret)

    local_secret = SigningKey.from_string(bytearray.fromhex(local_secret), curve=NIST256p)
    local_public = VerifyingKey.from_string(bytearray.fromhex(local_public), curve=NIST256p)

    ecdh = ECDH(curve=NIST256p, public_key=local_public, private_key=local_secret)
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


def receive_data_public_key() -> str:
    if SIMULATED_RECEIVE:
        _, data_public = generate_key_pair()
    else:
        raise NotImplemented  # TODO: implement

    return data_public


def receive_model_public_key() -> str:
    if SIMULATED_RECEIVE:
        _, model_public = generate_key_pair()
    else:
        raise NotImplemented  # TODO: implement

    return model_public


def receive_encrypted_data(data_shared_secret: str):
    if SIMULATED_RECEIVE:
        import __dummy
        __dummy.dummy_receive_encrypted_data(data_shared_secret)
    else:
        raise NotImplemented  # TODO: implement


def receive_encrypted_model(model_shared_secret: str):
    if SIMULATED_RECEIVE:
        import __dummy
        __dummy.dummy_receive_encrypted_model(model_shared_secret)
    else:
        raise NotImplemented  # TODO: implement
