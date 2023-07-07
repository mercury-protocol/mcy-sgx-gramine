from constants import QUOTE_PATH, REPORT_PATH,  GR_QUOTE, LOCAL_SECRET_KEY_PATH
from utils import read, write, generate_key_pair


def startup():
    secret, public = generate_key_pair()
    write(LOCAL_SECRET_KEY_PATH, secret)
    write(REPORT_PATH, public)
    quote = read(QUOTE_PATH, binary=True)
    write(GR_QUOTE, quote, binary=True)


if __name__ == "__main__":
    startup()
