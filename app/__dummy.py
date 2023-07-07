# TODO: remove this file once everything is properly implemented

from constants import LOCAL_SECRET_KEY_PATH
from utils import encrypt, decrypt, write, generate_key_pair, generate_shared_secret
import csv


def dummy_receive_encrypted_model(shared_secret, decrypted=False):
    with open("model.py", "w") as file:
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
    with open("model.py", "rb") as file:
        model = file.read()
        encrypted_model = encrypt(shared_secret, model)
    with open('model.py', 'wb') as file:
        file.write(encrypted_model)

    if decrypted:
        with open("model.py", "rb") as file:
            encrypted_model = file.read()
            model = decrypt(shared_secret, encrypted_model)
        with open('model.py', 'wb') as file:
            file.write(model)


def dummy_receive_encrypted_data(shared_secret, decrypted=False):
    with open("data.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])
        for i in range(10):
            writer.writerow([i, i])
    with open("data.csv", "rb") as file:
        data = file.read()
        encrypted_data = encrypt(shared_secret, data)
    with open('data.csv', 'wb') as file:
        file.write(encrypted_data)

    if decrypted:
        with open("data.csv", "rb") as file:
            encrypted_data = file.read()
            data = decrypt(shared_secret, encrypted_data)
        with open('data.csv', 'wb') as file:
            file.write(data)


def __simulate():
    from io import StringIO
    import pickle

    secret, public = generate_key_pair()
    write(LOCAL_SECRET_KEY_PATH, secret)

    _, data_public = generate_key_pair()
    _, model_public = generate_key_pair()
    shared_secret_data = generate_shared_secret(data_public)
    shared_secret_model = generate_shared_secret(model_public)

    dummy_receive_encrypted_model(shared_secret_model)
    dummy_receive_encrypted_data(shared_secret_data)

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
    model_coef = trained_model.coef_
    encrypted_trained_model = encrypt(shared_secret_model, pickle.dumps(trained_model))

    with open("trained_model.pkl", "wb") as file:
        pickle.dump(encrypted_trained_model, file)

    # delete model, unpickle encrypted model and decrypt it
    del trained_model
    del encrypted_trained_model
    with open("trained_model.pkl", "rb") as file:
        encrypted_trained_model = pickle.load(file)
        trained_model = pickle.loads(decrypt(shared_secret_model, encrypted_trained_model))
    assert model_coef == trained_model.coef_


def __simulate_light():
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from io import StringIO
    import pickle

    secret, public = generate_key_pair()
    write(LOCAL_SECRET_KEY_PATH, secret)

    _, model_public = generate_key_pair()
    shared_secret_model = generate_shared_secret(model_public)

    dummy_receive_encrypted_data(shared_secret_model, decrypted=True)

    with open("data.csv", "r") as file:
        data = file.read()
        data = StringIO(data)
        data = np.loadtxt(data, delimiter=",", dtype=int, skiprows=1).transpose()

    x = data[0].reshape((-1, 1))
    y = data[1]

    model = LinearRegression()
    model.fit(x, y)

    model_coef = model.coef_

    # pickle model to file
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

    # delete model and read it from the file
    del model
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    assert model_coef == model.coef_

    # pickle encrypted model
    encrypted_model = encrypt(shared_secret_model, pickle.dumps(model))
    with open("model.pkl", "wb") as file:
        pickle.dump(encrypted_model, file)

    # delete model, unpickle encrypted model and decrypt it
    del model
    with open("model.pkl", "rb") as file:
        encrypted_model = pickle.load(file)
        model = pickle.loads(decrypt(shared_secret_model, encrypted_model))
    assert model_coef == model.coef_


def __full_dummy_receive():
    secret, public = generate_key_pair()
    write(LOCAL_SECRET_KEY_PATH, secret)

    _, data_public = generate_key_pair()
    _, model_public = generate_key_pair()
    shared_secret_data = generate_shared_secret(data_public)
    shared_secret_model = generate_shared_secret(model_public)

    dummy_receive_encrypted_model(shared_secret_model, decrypted=True)
    dummy_receive_encrypted_data(shared_secret_data, decrypted=True)


if __name__ == '__main__':
    # __full_dummy_receive()
    __simulate()
    # __simulate_light()
