import os
from ecdsa.curves import NIST256p
from ecdsa.ecdh import ECDH
from ecdsa.keys import VerifyingKey


def load_remote_public_key():
    remote_public_key_hex = read("public_go")
    return VerifyingKey.from_string(bytearray.fromhex(remote_public_key_hex), curve=NIST256p)


def generate_key_pair():
    ecdh = ECDH(curve=NIST256p)
    ecdh.generate_private_key()
    secret = ecdh.private_key
    public = ecdh.get_public_key()

    return secret, public


def write(path, data):
    with open(path, "w") as f:
        f.write(data)


def read(path):
    with open(path, "r") as f:
        return f.read()


def main():
    secret, public = generate_key_pair()
    write("public_python", public.to_string().hex())

    os.system("go run dhke.go")
    remote_pub_key = load_remote_public_key()

    ecdh = ECDH(curve=NIST256p, public_key=public, private_key=secret)
    ecdh.load_received_public_key(remote_pub_key)

    shared_secret_hex = ecdh.generate_sharedsecret_bytes().hex()
    write("shared_secret_python", shared_secret_hex)

    shared_secret_hex_go = read("shared_secret_go")
    assert shared_secret_hex == shared_secret_hex_go


if __name__ == "__main__":
    main()
