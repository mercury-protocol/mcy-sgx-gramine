REPORT_PATH = "/dev/attestation/user_report_data"
QUOTE_PATH = "/dev/attestation/quote"


def write_report():
    with open(REPORT_PATH, "w") as f:
        f.write("testreport")


def get_quote():
    with open(QUOTE_PATH, "r") as f:
        return f.read()


write_report()
print(get_quote())
