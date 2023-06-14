from constants import OrderSide, SHARED_SECRET_DATA, SHARED_SECRET_MODEL
from utils import generate_key_pair, generate_shared_secret, write


def train_model(order_id: str, side: OrderSide):
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
    __dummy.dummy_receive_encrypted_data()
    __dummy.dummy_receive_encrypted_model()

    # decrypt data and model


if __name__ == '__main__':
    train_model('1', OrderSide('buy'))
