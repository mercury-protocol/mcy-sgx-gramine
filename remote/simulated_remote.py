import json

from remote.utils import encrypt, write, read, generate_key_pair, derive_shared_secret, wait_file, load_trained_model


DATA_PATH = "data.csv"
MODEL_PATH = "model.py"

IAS_REPORT = "../app/io_files/ias.report"
IAS_SIGNATURE = "../app/io_files/ias.sig"
IAS_CERTIFICATE = "../app/io_files/ias.cert"

DATA_PUBLIC_KEY_PATH = "../app/io_files/data_public_key"
MODEL_PUBLIC_KEY_PATH = "../app/io_files/model_public_key"
REMOTE_PUBLIC_KEY_PATH = "../app/io_files/local_public_key"
ATTESTATION_REPORT_PATH = "../app/io_files/attestation_report.json"
ENCRYPTED_DATA_PATH = "../app/io_files/encrypted_data.csv"
ENCRYPTED_MODEL_PATH = "../app/io_files/encrypted_model.py"
ENCRYPTED_TRAINED_MODEL_PATH = "../app/io_files/encrypted_trained_model.pkl"


def check_attestation():
    # TODO: GROUP_OUT_OF_DATE should be fixed in attestation and not be accepted here
    wait_file(IAS_REPORT)

    attestation_report = {
        "X-IASReport-Signature": read(IAS_SIGNATURE),
        "X-IASReport-Signing-Certificate": read(IAS_CERTIFICATE),
        "Body": json.loads(read(IAS_REPORT))
    }
    assert attestation_report["Body"]["isvEnclaveQuoteStatus"] in ["OK", "GROUP_OUT_OF_DATE"]


def send_encrypted_model(shared_secret: str):
    with open(MODEL_PATH, "rb") as file:
        model = file.read()
        encrypted_model = encrypt(shared_secret, model)
    with open(ENCRYPTED_MODEL_PATH, 'wb') as file:
        file.write(encrypted_model)


def send_encrypted_data(shared_secret: str):
    with open(DATA_PATH, "rb") as file:
        data = file.read()
        encrypted_data = encrypt(shared_secret, data)
    with open(ENCRYPTED_DATA_PATH, 'wb') as file:
        file.write(encrypted_data)


def train_model():
    from remote.model import run
    return run(DATA_PATH)


def simulate():
    check_attestation()

    secret, public = generate_key_pair()
    write(DATA_PUBLIC_KEY_PATH, public)
    write(MODEL_PUBLIC_KEY_PATH, public)

    remote_public = read(REMOTE_PUBLIC_KEY_PATH)
    shared_secret = derive_shared_secret(secret, public, remote_public)

    send_encrypted_model(shared_secret)
    send_encrypted_data(shared_secret)

    wait_file(ENCRYPTED_TRAINED_MODEL_PATH)
    trained_model = load_trained_model(ENCRYPTED_TRAINED_MODEL_PATH, shared_secret)
    expected_trained_model = train_model()

    assert trained_model.coef_ == expected_trained_model.coef_
    print("Model training was successful!")


if __name__ == "__main__":
    simulate()
