from constants import QUOTE_PATH, REPORT_PATH,  GR_QUOTE, LOCAL_SECRET_KEY_PATH
from utils import read, write, generate_key_pair, derive_public_from_secret


def save_local_secret():
    secret, public = generate_key_pair()
    write(LOCAL_SECRET_KEY_PATH, secret)


def save_quote():
    secret = read(LOCAL_SECRET_KEY_PATH)
    public = derive_public_from_secret(secret)
    write(REPORT_PATH, public)
    quote = read(QUOTE_PATH, binary=True)
    write(GR_QUOTE, quote, binary=True)


if __name__ == "__main__":
    save_local_secret()
    save_quote()
