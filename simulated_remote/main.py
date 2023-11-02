import csv
import os
import pickle
import time
from io import StringIO
from ecdsa.keys import SigningKey, VerifyingKey
from ecdsa.curves import NIST256p
from ecdsa.ecdh import ECDH

from app.trusted_src.utils import encrypt, decrypt, write, read, generate_key_pair


DATA_PUBLIC_KEY_PATH = "../app/data_public_key"
MODEL_PUBLIC_KEY_PATH = "../app/model_public_key"
REMOTE_PUBLIC_KEY_PATH = "../app/local_public_key"
ATTESTATION_REPORT_PATH = "../app/attestation_report.json"
ENCRYPTED_DATA_PATH = "../app/encrypted_data.csv"
ENCRYPTED_MODEL_PATH = "../app/encrypted_model.py"
ENCRYPTED_TRAINED_MODEL_PATH = "../app/encrypted_trained_model.pkl"


def derive_shared_secret(secret: str, public: str, remote_public: str) -> str:
    secret = SigningKey.from_string(bytearray.fromhex(secret), curve=NIST256p)
    public = VerifyingKey.from_string(bytearray.fromhex(public), curve=NIST256p)

    ecdh = ECDH(curve=NIST256p, public_key=public, private_key=secret)
    ecdh.load_received_public_key_bytes(bytearray.fromhex(remote_public))

    shared_secret = ecdh.generate_sharedsecret_bytes().hex()

    return shared_secret


def send_encrypted_model(shared_secret: str):
    with open(ENCRYPTED_MODEL_PATH, "w") as file:
        file.write(
            """print('start dummy model training')

            def run(data):
                import numpy as np
                from sklearn.linear_model import LinearRegression

                data = np.loadtxt(data, delimiter=",", dtype=int, skiprows=1).transpose()
                x = data[0].reshape((-1, 1))
                y = data[1]

                model = LinearRegression()
                model.fit(x, y)

                return model
            """
        )
    with open(ENCRYPTED_MODEL_PATH, "rb") as file:
        model = file.read()
        encrypted_model = encrypt(shared_secret, model)
    with open(ENCRYPTED_MODEL_PATH, 'wb') as file:
        file.write(encrypted_model)


def send_encrypted_data(shared_secret: str):
    with open(ENCRYPTED_DATA_PATH, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])
        for i in range(10):
            writer.writerow([i, i])
    with open(ENCRYPTED_DATA_PATH, "rb") as file:
        data = file.read()
        encrypted_data = encrypt(shared_secret, data)
    with open(ENCRYPTED_DATA_PATH, 'wb') as file:
        file.write(encrypted_data)


def get_expected_model(shared_secret: str):
    with open(ENCRYPTED_DATA_PATH, "rb") as file:
        encrypted_data = file.read()
    data = decrypt(shared_secret, encrypted_data).decode("utf-8")
    data = StringIO(data)

    with open(ENCRYPTED_MODEL_PATH, "rb") as file:
        encrypted_model = file.read()
        model = decrypt(shared_secret, encrypted_model)

    exec(model)
    trained_model = eval("run(data)")
    return trained_model


def simulate():
    # TODO: GROUP_OUT_OF_DATE should be fixed in attestation and not be accepted here
    attestation_report = read(ATTESTATION_REPORT_PATH)
    assert attestation_report["Body"]["isvEnclaveQuoteStatus"] in ["OK", "GROUP_OUT_OF_DATE"]

    secret, public = generate_key_pair()
    write(DATA_PUBLIC_KEY_PATH, secret)
    write(MODEL_PUBLIC_KEY_PATH, secret)

    remote_public = read(REMOTE_PUBLIC_KEY_PATH)
    shared_secret = derive_shared_secret(secret, public, remote_public)

    send_encrypted_model(shared_secret)
    send_encrypted_data(shared_secret)

    while not os.path.exists(ENCRYPTED_TRAINED_MODEL_PATH):
        print("Encrypted trained model not received yet, waiting...")
        time.sleep(5)

    with open(ENCRYPTED_TRAINED_MODEL_PATH, "rb") as file:
        encrypted_trained_model = pickle.load(file)
        trained_model = pickle.loads(decrypt(shared_secret, encrypted_trained_model))

    expected_trained_model = get_expected_model(shared_secret)

    assert trained_model.coef_ == expected_trained_model.coef_


if __name__ == "__main__":
    simulate()
