# mcy-sgx-gramine

## How to run model training in docker:
* build and run the container with the following command:
  * `sudo docker-compose up app`
* you have to use sudo, because `/dev/sgx_enclave` and `/dev/sgx_provision`
devices have to be passed for the container and that needs root privileges
* note, that if you use docker-compose with sudo, you won't see the image nor the container in docker desktop
  * list containers: `sudo docker ps -a`
  * delete all containers: `sudo docker rm $(sudo docker ps -a -q)`
  * list all images: `sudo docker images`
  * delete all docker images: `sudo docker rmi $(sudo docker images -q)`

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


---
_Note: if you are using virtual environment for local run, you have to add this to the top of ra.py:_<br>
`import sys`<br>
`sys.path.append("<VENV PATH>/lib/python<VERSION>/site-packages")`<br>
_example:_<br>
`"/home/mercury/Documents/repos/mcy-sgx-gramine/venv/lib/python3.8/site-packages"`