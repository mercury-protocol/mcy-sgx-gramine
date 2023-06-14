# TODO: remove this file once everything is properly implemented

from constants import SHARED_SECRET_DATA, SHARED_SECRET_MODEL
from ecdsa.keys import SigningKey, VerifyingKey
from ecdsa.curves import NIST256p
from ecdsa.ecdh import ECDH
from utils import encrypt, decrypt, read, generate_key_pair, generate_shared_secret
import csv


def generate_shared_secret_dummy(remote_public_key: str):
    secret, public = generate_key_pair()

    secret = SigningKey.from_string(bytearray.fromhex(secret), curve=NIST256p)
    public = VerifyingKey.from_string(bytearray.fromhex(public), curve=NIST256p)

    ecdh = ECDH(curve=NIST256p, public_key=public, private_key=secret)
    ecdh.load_received_public_key_bytes(bytearray.fromhex(remote_public_key))

    shared_secret = ecdh.generate_sharedsecret_bytes().hex()

    return shared_secret


def dummy_receive_encrypted_model():
    # _, remote_public_dummy = generate_key_pair()
    # shared_secret = generate_shared_secret_dummy(remote_public_dummy)
    shared_secret = read(SHARED_SECRET_MODEL)

    with open("model.py", "w") as file:
        file.write("print('hello model')\n")
    with open("model.py", "rb") as file:
        model = file.read()
        encrypted_model = encrypt(shared_secret, model)
    with open('model.py', 'wb') as file:
        file.write(encrypted_model)

    # to_decrypt = input("model encrypted. decrypt? (y/n)").lower() == "y"
    # if to_decrypt:
    #     with open("model.py", "rb") as file:
    #         encrypted_model = file.read()
    #         model = decrypt(shared_secret, encrypted_model)
    #     with open('model.py', 'wb') as file:
    #         file.write(model)


def dummy_receive_encrypted_data():
    # _, remote_public_dummy = generate_key_pair()
    # shared_secret = generate_shared_secret_dummy(remote_public_dummy)
    shared_secret = read(SHARED_SECRET_DATA)

    with open("data.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow([0, 1, 2])
    with open("data.csv", "rb") as file:
        data = file.read()
        encrypted_data = encrypt(shared_secret, data)
    with open('data.csv', 'wb') as file:
        file.write(encrypted_data)

    # to_decrypt = input("data encrypted. decrypt? (y/n)").lower() == "y"
    # if to_decrypt:
    #     with open("data.csv", "rb") as file:
    #         encrypted_data = file.read()
    #         data = decrypt(shared_secret, encrypted_data)
    #     with open('data.csv', 'wb') as file:
    #         file.write(data)

