import json
import csv
import os
import pickle
import time
from io import StringIO

from remote.utils import encrypt, decrypt, write, read, generate_key_pair, derive_shared_secret


IAS_REPORT = "../app/ias.report"
IAS_SIGNATURE = "../app/ias.sig"
IAS_CERTIFICATE = "../app/ias.cert"

DATA_PUBLIC_KEY_PATH = "../app/data_public_key"
MODEL_PUBLIC_KEY_PATH = "../app/model_public_key"
REMOTE_PUBLIC_KEY_PATH = "../app/local_public_key"
ATTESTATION_REPORT_PATH = "../app/attestation_report.json"
ENCRYPTED_DATA_PATH = "../app/encrypted_data.csv"
ENCRYPTED_MODEL_PATH = "../app/encrypted_model.py"
ENCRYPTED_TRAINED_MODEL_PATH = "../app/encrypted_trained_model.pkl"


def check_attestation():
    # TODO: GROUP_OUT_OF_DATE should be fixed in attestation and not be accepted here
    attestation_report = {
        "X-IASReport-Signature": read(IAS_SIGNATURE),
        "X-IASReport-Signing-Certificate": read(IAS_CERTIFICATE),
        "Body": json.loads(read(IAS_REPORT))
    }
    assert attestation_report["Body"]["isvEnclaveQuoteStatus"] in ["OK", "GROUP_OUT_OF_DATE"]


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
    check_attestation()

    secret, public = generate_key_pair()
    write(DATA_PUBLIC_KEY_PATH, public)
    write(MODEL_PUBLIC_KEY_PATH, public)

    remote_public = read(REMOTE_PUBLIC_KEY_PATH)
    shared_secret = derive_shared_secret(secret, public, remote_public)

    send_encrypted_model(shared_secret)
    send_encrypted_data(shared_secret)

    while not os.path.exists(ENCRYPTED_TRAINED_MODEL_PATH):
        print("Encrypted trained model not received yet, waiting...")
        time.sleep(5)
    print("Encrypted trained model received")

    with open(ENCRYPTED_TRAINED_MODEL_PATH, "rb") as file:
        encrypted_trained_model = pickle.load(file)
        trained_model = pickle.loads(decrypt(shared_secret, encrypted_trained_model))

    expected_trained_model = get_expected_model(shared_secret)

    assert trained_model.coef_ == expected_trained_model.coef_
    print("Model training was successful!")


if __name__ == "__main__":
    simulate()
