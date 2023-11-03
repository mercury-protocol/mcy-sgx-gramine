import pickle
from io import StringIO

from sgx_constants import DATA_PUBLIC_KEY_PATH, MODEL_PUBLIC_KEY_PATH, ENCRYPTED_TRAINED_MODEL_PATH
from sgx_utils import receive_public_key, receive_data, receive_model, derive_shared_secret, encrypt


def train_model():
    data_public = receive_public_key(DATA_PUBLIC_KEY_PATH)
    model_public = receive_public_key(MODEL_PUBLIC_KEY_PATH)

    data_shared_secret = derive_shared_secret(data_public)
    model_shared_secret = derive_shared_secret(model_public)

    data = receive_data(data_shared_secret)
    model = receive_model(model_shared_secret)

    # TODO: implement data format verification
    # TODO: implement model format verification

    exec(model)
    data = StringIO(data)
    trained_model = eval("run(data)")
    del data
    # TODO: implement smart check that data doesn't go out
    encrypted_trained_model = encrypt(model_shared_secret, pickle.dumps(trained_model))
    del trained_model

    with open(ENCRYPTED_TRAINED_MODEL_PATH, "wb") as file:
        pickle.dump(encrypted_trained_model, file)
