---
paths:
  - "remote/**"
---

# SGX Remote Client (Simulated)

The untrusted counterpart to `app/`. Plays both the data-owner and model-owner roles for local end-to-end testing: verifies the enclave's IAS report, runs ECDH with the enclave's public key, encrypts data and model into the enclave's input dir, waits for the encrypted trained model, decrypts it, and compares against a locally-trained baseline.

## Design intent

- **Protocol mirror of `app/`.** Constants, key sizes, curve (NIST256p), and Fernet usage must match `app/sgx_utils.py` byte-for-byte. Any change to the enclave-side protocol requires the matching change here in the same commit.
- **Filesystem-coupled to the enclave.** `constants.py` paths point into `../app/input/` and `../app/output/`; this script assumes both processes share a working directory layout. This is the test harness, not a production client.
- **Trains a baseline locally to validate correctness.** `simulate_remote` asserts the enclave-returned model matches a locally-trained model on the same data — the round-trip is the test.

## Patterns

- `utils.py` here is a deliberately separate copy from `app/sgx_utils.py` because the remote side runs outside the enclave and has different file paths and a simpler key-loading flow. Don't merge them — the duplication encodes the trust boundary.
- The model under test is whatever `remote/model.py` defines via a top-level `run(data)` function; the enclave receives the source and `exec`s it.

## Gotchas

- Accepts `GROUP_OUT_OF_DATE` in `check_attestation` (paired with the same TODO on the enclave side). Don't tighten one side without the other.
- Sends the model as Python source, not as serialised weights — the enclave compiles it. If you swap to weights, the enclave's `sgx_train_model.py` must change too.
