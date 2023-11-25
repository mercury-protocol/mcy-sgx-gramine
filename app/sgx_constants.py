# gramine pseudo filesystem paths
REPORT_PATH = "/dev/attestation/user_report_data"
QUOTE_PATH = "/dev/attestation/quote"

# encrypted files
LOCAL_SECRET_KEY_PATH = "local_secret_key"

# I/O files
DATA_PUBLIC_KEY_PATH = "input/data_public_key"
MODEL_PUBLIC_KEY_PATH = "input/model_public_key"
ENCRYPTED_DATA_PATH = "input/encrypted_data.csv"
ENCRYPTED_MODEL_PATH = "input/encrypted_model.py"
LOCAL_PUBLIC_KEY_PATH = "output/local_public_key"
ENCRYPTED_TRAINED_MODEL_PATH = "output/encrypted_trained_model.pkl"

# gramine I/O files
GR_QUOTE = "output/gr.quote"
IAS_REPORT = "output/ias.report"
IAS_SIGNATURE = "output/ias.sig"
IAS_CERTIFICATE = "output/ias.cert"
