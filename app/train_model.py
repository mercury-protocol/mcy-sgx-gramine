import pickle

from io import StringIO

from constants import DATA_SHARED_SECRET_KEY_PATH, MODEL_SHARED_SECRET_KEY_PATH
from utils import (receive_data_public_key, receive_model_public_key, receive_encrypted_data, receive_encrypted_model,
                   derive_shared_secret, write, decrypt, encrypt)


def train_model():
    data_public = receive_data_public_key()
    model_public = receive_model_public_key()

    # generate the shared secrets and save them
    data_shared_secret = derive_shared_secret(data_public)
    write(DATA_SHARED_SECRET_KEY_PATH, data_shared_secret)
    model_shared_secret = derive_shared_secret(model_public)
    write(MODEL_SHARED_SECRET_KEY_PATH, model_shared_secret)

    # receive encrypted data and model
    receive_encrypted_data(data_shared_secret)
    receive_encrypted_model(model_shared_secret)

    # decrypt data and model
    with open("data.csv", "rb") as file:
        encrypted_data = file.read()
    data = decrypt(data_shared_secret, encrypted_data).decode("utf-8")
    del encrypted_data
    data = StringIO(data)

    with open("model.py", "rb") as file:
        encrypted_model = file.read()
        model = decrypt(model_shared_secret, encrypted_model)
    del encrypted_model

    exec(model)
    trained_model = eval("run(data)")
    # TODO: implement smart check that data doesn't go out
    encrypted_trained_model = encrypt(model_shared_secret, pickle.dumps(trained_model))

    with open("trained_model.pkl", "wb") as file:
        pickle.dump(encrypted_trained_model, file)


if __name__ == "__main__":
    train_model()
