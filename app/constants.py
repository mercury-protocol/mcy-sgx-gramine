# gramine pseudo filesystem paths
REPORT_PATH = "/dev/attestation/user_report_data"
QUOTE_PATH = "/dev/attestation/quote"

# encrypted files
LOCAL_SECRET_KEY_PATH = "local_secret_key"

# I/O files
LOCAL_PUBLIC_KEY_PATH = "local_public_key"
ATTESTATION_REPORT_PATH = "attestation_report.json"
DATA_PUBLIC_KEY_PATH = "data_public_key"
MODEL_PUBLIC_KEY_PATH = "model_public_key"
ENCRYPTED_DATA_PATH = "encrypted_data.csv"
ENCRYPTED_MODEL_PATH = "encrypted_model.py"
ENCRYPTED_TRAINED_MODEL_PATH = "encrypted_trained_model.pkl"

# gramine output files
GR_QUOTE = "gr.quote"
IAS_REPORT = "ias.report"
IAS_SIGNATURE = "ias.sig"
IAS_CERTIFICATE = "ias.cert"

IAS_API_KEY = "909c02172d964178a794d055e98c41d8"  # TODO: store it securely

SIMULATED_RECEIVE = True
