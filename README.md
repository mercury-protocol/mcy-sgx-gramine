# mcy-sgx-gramine

Note: if you are using virtual environment for local run, you have to add this to the top of ra.py:

`import sys`

`sys.path.append("<VENV PATH>/lib/python<VERSION>/site-packages")`

example:
`"/home/mercury/Documents/repos/mcy-sgx-gramine/venv/lib/python3.8/site-packages"`

## How to decrypt trained model:
    from base64 import b64encode
    from cryptography.fernet import Fernet
    import pickle
    
    def decrypt(shared_secret: str, encrypted: bytes) -> bytes:
        shared_secret = b64encode(bytearray.fromhex(shared_secret))
        fernet = Fernet(shared_secret)
        decrypted = fernet.decrypt(encrypted)
        return decrypted
    
    with open("model.pkl", "rb") as file:
        encrypted_model = pickle.load(file)
        model = pickle.loads(decrypt(shared_secret_model, encrypted_model))
