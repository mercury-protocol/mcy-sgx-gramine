from ecdsa import ECDH, NIST256p


def test_dhke():
    """Diffie-Hellman ECDSA Key Exchange using the P-256 curve"""
    # Generate local key pair
    local = ECDH(curve=NIST256p)
    local.generate_private_key()
    local_public_key = local.get_public_key().to_string()
    with open("local_public_key", "wb") as f:
        f.write(local_public_key)

    # Generate remote key pair
    remote = ECDH(curve=NIST256p)
    remote.generate_private_key()
    remote_public_key = remote.get_public_key().to_string()
    with open("remote_public_key", "wb") as f:
        f.write(remote_public_key)

    # Load remote public key and generate shared secret
    with open("remote_public_key", "rb") as f:
        remote_public_key_received = f.read()
    local.load_received_public_key_bytes(remote_public_key_received)
    local_shared_secret = local.generate_sharedsecret_bytes()

    # Load local public key on remote and generate shared secret
    with open("local_public_key", "rb") as f:
        local_public_key_received = f.read()
    remote.load_received_public_key_bytes(local_public_key_received)
    remote_shared_secret = remote.generate_sharedsecret_bytes()

    print()
    print(f"local_public_key ({len(local_public_key)} bytes):", local_public_key.hex())
    print(f"local_shared_secret ({len(local_shared_secret)} bytes):", local_shared_secret.hex())

    assert local_shared_secret == remote_shared_secret
    assert local_public_key == local_public_key_received
    assert remote_public_key == remote_public_key_received
    assert local_public_key != remote_public_key

    # Repeat test with hex keys - only on local side
    with open("remote_public_key_hex", "w") as f:
        f.write(remote_public_key.hex())
    with open("remote_public_key_hex", "r") as f:
        remote_public_key_received = bytearray.fromhex(f.read())
    local.load_received_public_key_bytes(remote_public_key_received)
    local_shared_secret = local.generate_sharedsecret_bytes()
    assert local_shared_secret == remote_shared_secret
