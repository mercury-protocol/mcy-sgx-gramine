import pickle
from io import StringIO
from constants import SHARED_SECRET_DATA, SHARED_SECRET_MODEL
from utils import generate_key_pair, generate_shared_secret, write, decrypt, encrypt


def train_model(order_id: str):
    # receive public keys
    # TODO: implement properly
    _, data_public = generate_key_pair()
    _, model_public = generate_key_pair()

    # generate the shared secrets and save them
    shared_secret_data = generate_shared_secret(data_public)
    write(SHARED_SECRET_DATA, shared_secret_data)
    shared_secret_model = generate_shared_secret(model_public)
    write(SHARED_SECRET_MODEL, shared_secret_model)

    # receive encrypted data and model
    # TODO: implement properly
    import __dummy
    __dummy.dummy_receive_encrypted_data(shared_secret_data)
    __dummy.dummy_receive_encrypted_model(shared_secret_model)

    # decrypt data and model
    with open("data.csv", "rb") as file:
        encrypted_data = file.read()
    data = decrypt(shared_secret_data, encrypted_data).decode("utf-8")
    data = StringIO(data)

    with open("model.py", "rb") as file:
        encrypted_model = file.read()
        model = decrypt(shared_secret_model, encrypted_model)

    exec(model)
    trained_model = eval("run(data)")
    # TODO: implement smart check that data doesn't go out
    encrypted_trained_model = encrypt(shared_secret_model, pickle.dumps(trained_model))

    with open("trained_model.pkl", "wb") as file:
        pickle.dump(encrypted_trained_model, file)


if __name__ == "__main__":
    train_model("dummy_order_id")
