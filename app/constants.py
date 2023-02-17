# gramine pseudo filesystem paths
REPORT_PATH = "/dev/attestation/user_report_data"
QUOTE_PATH = "/dev/attestation/quote"
KEY_NAME = "ecdsa_secret"
KEY_PATH = f"/dev/attestation/keys/{KEY_NAME}"

# gramine output files
GR_QUOTE = "gr.quote"
IAS_REPORT = "ias.report"
IAS_SIGNATURE = "ias.sig"
IAS_CERTIFICATE = "ias.cert"

IAS_API_KEY = "909c02172d964178a794d055e98c41d8"  # TODO: store it securely
