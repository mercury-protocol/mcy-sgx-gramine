import os

commands = [
    "make clean",
    "make SGX=1",
    "gramine-sgx helloworld"
]

if __name__ == "__main__":
    os.chdir("helloworld")
    for cmd in commands:
        os.system(cmd)
