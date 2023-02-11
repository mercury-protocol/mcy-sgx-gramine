REPORT_PATH = "/dev/attestation/user_report_data"
QUOTE_PATH = "/dev/attestation/quote"
KEY_NAME = "ecdsa_pk"
KEY_PATH = f"/dev/attestation/keys/{KEY_NAME}"


def generate_key_pair():
    pass


def save_private_key(private_key):
    with open(KEY_PATH, "w") as f:
        f.write(private_key)


def write_report(report):
    with open(REPORT_PATH, "w") as f:
        f.write(report)


def get_quote():
    with open(QUOTE_PATH, "r") as f:
        return f.read()


write_report()
print(get_quote())
