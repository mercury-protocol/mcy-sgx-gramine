import os

commands = [
  # TODO: add commands
]

if __name__ == "__main__":
    os.chdir("ra-tls-secret-prov")
    for cmd in commands:
        os.system(cmd)
