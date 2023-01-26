import os

commands = [
  # TODO: add commands
]

if __name__ == "__main__":
    os.chdir("ra-tls-mbedtls")
    for cmd in commands:
        os.system(cmd)
