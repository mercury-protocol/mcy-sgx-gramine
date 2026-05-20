---
paths:
  - "app/**"
---

# SGX Enclave App

Python application that runs inside an Intel SGX enclave via Gramine. Its job: prove to a remote party that it is genuinely executing in an enclave (remote attestation), establish encrypted channels for the user's data and model, run the model on the data, and return the encrypted result. The enclave is the only place the data and model exist in plaintext.

## Design intent

- **The enclave is the trust boundary.** Anything written to `output/` is visible to the untrusted host; anything in `input/` arrived from the host. Only `local_secret_key` is encrypted-at-rest (manifest `type = "encrypted"`, sealed to `_sgx_mrenclave`). Treat every other file as adversary-readable.
- **Attestation gates everything.** `sgx_main.py` blocks on `IAS_REPORT` before `train_model()` runs. The remote party fetches that report and verifies it before sending data. Do not move the gate.
- **Two independent channels.** Data and model each get their own ECDH(NIST256p) handshake and Fernet key derived from the shared secret. Don't collapse them — the threat model assumes the data owner and model owner may be different parties.
- **EPID, not DCAP.** The manifest pins `sgx.remote_attestation = "epid"` and uses Intel's IAS service. Switching to DCAP is a protocol change that affects `attestation.py`, the manifest, and the remote client.

## Patterns

- Plaintext I/O uses `sgx_utils.read`/`write`; encrypted channel I/O uses `receive_data`/`receive_model` which wrap `wait_file` + `decrypt`. Don't open files directly in the role code.
- Long-lived secrets (the enclave's ECDH private key) live only at `local_secret_key`, which is the one file mounted as `type = "encrypted"`. New secrets must be added to the manifest the same way or they leak to the host.
- Filenames are centralised in `sgx_constants.py` and must match `sgxapp.manifest.template`'s `trusted_files` / `allowed_files`. Adding a file means updating both.

## Conventions

- The enclave entrypoint is `sgx_main.py`, launched by `loader.argv` in the manifest. `main.py` is the *host-side* launcher — it builds the enclave and starts `attestation.py` in a sibling thread. Don't confuse the two.
- `env.py` reads `IAS_API_KEY` and `RA_CLIENT_SPID` from the host environment; these never enter the enclave.

## Gotchas

- `sgx.allowed_files` is a known weakness (see TODO in the manifest) — files listed there are visible to the host with no integrity check. Adding to that list expands the attack surface; prefer `trusted_files` (integrity-checked at load) or encrypted mounts.
- `sgx_train_model.py` uses `exec`/`eval` on the decrypted user model. That's only acceptable because the model arrived over an authenticated encrypted channel from a party we already attested to — don't reuse this pattern for any other input.
- `attestation.py` currently accepts `GROUP_OUT_OF_DATE` (see TODO). Tightening that check is a coordinated change with `remote/`.
