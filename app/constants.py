# gramine pseudo filesystem paths
REPORT_PATH = "/dev/attestation/user_report_data"
QUOTE_PATH = "/dev/attestation/quote"

# encrypted files
LOCAL_SECRET_KEY_PATH = "local_secret_key"
MODEL_SHARED_SECRET_KEY_PATH = "model_shared_secret_key"
DATA_SHARED_SECRET_KEY_PATH = "data_shared_secret_key"

# I/O files
DATA_PUBLIC_KEY_PATH = "data_public_key"
MODEL_PUBLIC_KEY_PATH = "model_public_key"
ENCRYPTED_DATA_PATH = "data.csv"
ENCRYPTED_MODEL_PATH = "model.py"

# gramine output files
GR_QUOTE = "gr.quote"
IAS_REPORT = "ias.report"
IAS_SIGNATURE = "ias.sig"
IAS_CERTIFICATE = "ias.cert"

IAS_API_KEY = "909c02172d964178a794d055e98c41d8"  # TODO: store it securely

SIMULATED_RECEIVE = True
