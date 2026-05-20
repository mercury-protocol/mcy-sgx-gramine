# mcy-sgx-gramine

Confidential and distributed training building blocks for the [Mercury Protocol](https://mercuryprotocol.netlify.app) off-chain node.

## Context

Mercury is a decentralized peer-to-peer GPU network for training AI models, designed to give developers cheaper and privacy-preserving compute by tapping the world's dormant CPUs and GPUs. The off-chain node has three roles:

- **Watcher** — picks up work requests from the blockchain, estimates compute requirements, selects nodes, and posts the resulting attestations back on-chain. Lives upstream in [Vulkan](https://github.com/mercury-protocol/vulkan) and the smart-contract layer.
- **Leader** — aggregates gradients from the workers each step and broadcasts the updated state dict back.
- **Worker** — trains the user's model on its shard of the data and emits gradients.

This repository contains two of the foundational pieces of the off-chain node:

- **`mcy_dist_ai/`** — the leader and worker, published as a PyPI package. Drops into the Vulkan transport layer; gives Mercury its synchronous data-parallel training.
- **`app/` + `remote/`** — a Gramine SGX enclave application that proves out the attestation and encrypted-channel flow underpinning Mercury's verifiable-compute story. In production, attestations produced inside the enclave are forwarded by the leader to the watcher, which posts them on-chain.

Both subsystems are MVPs of their respective layers; the larger integration (workers running inside enclaves, attesting to a GPU TEE) is described in the Mercury Litepaper.

---

## Architecture

### Distributed training (`mcy_dist_ai/`)

Each process is launched as either a `LEADER` or a `WORKER`. Workers train batches on local data shards and emit gradients; the leader averages them, applies the optimizer step, and broadcasts the new state dict back. Synchronous stochastic gradient descent over file-based inter-process communication.

```
WORKER 1 ──gradient_1.pth──┐
WORKER 2 ──gradient_2.pth──┼──► LEADER ──state_dict.pth──► all workers
WORKER N ──gradient_N.pth──┘     (mean grads → optimizer.step → broadcast)
```

The Vulkan transport ships these files between hosts; this package only assumes they arrive. A `<name>.pth` + `<name>_ready.pth` sentinel pair handles producer/consumer races without locks.

### Confidential model training (`app/`, `remote/`)

A Python application runs inside an Intel SGX enclave via Gramine. The remote party fetches the enclave's IAS attestation report and verifies it, then performs ECDH (NIST P-256) over two independent channels — one for the data, one for the model — and ships Fernet-encrypted payloads in. The enclave runs the model and returns the encrypted result.

```
DATA OWNER  ──encrypted data ──┐                  ┌── encrypted result
                               ├──► SGX ENCLAVE ──┤
MODEL OWNER ──encrypted model──┘  (attested via   └── (sealed to recipient)
                                   Intel IAS)
```

Two channels exist because Mercury's threat model treats the data owner and model owner as potentially distinct, mutually-distrustful parties — both shipping IP to the same untrusted compute provider.

---

## Notable engineering decisions

- **File-based IPC, not sockets.** Gramine's SGX backend constrains network APIs. Since the production target is workers running inside enclaves, all node coordination already uses files declared as enclave inputs and outputs.
- **Sentinel-file ready protocol.** Each transfer is two files (`x.pth` + `x_ready.pth`) so consumers never read a half-written tensor. The transport layer can copy in any order without coordination.
- **Plugin contract for user models.** The framework imports a user-supplied `user_script.py` dynamically and pip-installs the accompanying `user_requirements.txt` at startup. Users define their model, optimizer, data loader, and per-batch training function — the framework provides the distributed orchestration around them, with no DevOps for the user.
- **Manifest-enforced trust boundary.** Inside the enclave only the long-lived ECDH private key is sealed (`type = "encrypted"`, bound to MRENCLAVE). Every other file is host-readable by design; secrets only exist in memory while the enclave is running.
- **Two encryption channels, not one.** Data and model arrive over separate ECDH handshakes so the data owner and model owner can be independent parties on the Mercury marketplace, neither having to trust the other.
- **Attestation-gated execution.** Training does not start until the IAS report has been written; remote parties refuse to send their encrypted inputs until they have verified the report themselves. The same report is what the watcher will eventually upload on-chain as the verifiable-compute proof.
- **End-to-end tests run real code, not mocks.** A multi-process harness replaces the P2P transport with a local file shuttler and spawns the actual leader and worker processes; assertions are on the resulting model's accuracy.

---

## Install & use

### As a library

```bash
pip install mcy_dist_ai
```

Write a `user_script.py` exposing the required symbols — full contract in [docs/user_script_requirements.md](docs/user_script_requirements.md), worked example in [tests/examples/image_classifier/user_script.py](tests/examples/image_classifier/user_script.py) — plus a `user_requirements.txt` for its dependencies. The Vulkan layer then launches the leader and N workers on participating peers.

### `mcy-split-data` CLI

Pre-splits a dataset into N tensor partitions so heavy preprocessing runs once instead of per worker:

```
mcy-split-data <split_into> <data_path> <output_dir_path> <user_script_path>
```

Workers launched with `--tensor_load` consume these partitions directly.

### Running the SGX enclave

Requires SGX-capable hardware and Intel IAS credentials (`IAS_API_KEY`, `RA_CLIENT_SPID`) in a project-root `.env`. Full instructions in [docs/sgx_app.md](docs/sgx_app.md). In short:

```bash
sudo docker-compose up app                    # builds and runs the enclave
sudo python3 remote/simulated_remote.py       # plays the data + model owner
```

The `remote/` simulator trains a baseline locally and asserts the enclave-returned model matches it — the round-trip is the test.

---

## Layout

```
mcy_dist_ai/    Published Python package — leader, worker, user-script integration
app/            Enclave entrypoints, attestation, encrypted channels, Gramine manifest
remote/         Untrusted counterpart to app/, used for local end-to-end testing
tests/          Multi-process simulation harness and example user scripts
docs/           User-script contract, SGX flow, sample attestation reports
```

---

## Testing

```bash
pip install -r requirements_test.txt
pytest tests/
```

End-to-end tests run the real leader and worker code under a simulated network. Each node is launched in its own OS process via `multiprocessing`, with a role-specific argv and an isolated working directory under `tests/temp/`. Because the package resolves all coordination paths against the process `cwd` at import time, every child inhabits an independent filesystem namespace — the same isolation it would see on a separate host, with its own logger, its own user-script import, and its own copy of the per-node configuration. The Vulkan transport layer is replaced by an asynchronous file shuttler that copies gradient and state-dict files between those directories on the producer/consumer sentinel pattern the production transport uses.

The harness supports arbitrary worker counts and can run the network either fully in parallel or sequentially. Included tests train MNIST classifiers with 1, 2, and 4 workers and assert on classification accuracy; an LLM fine-tuning example is also included as a work-in-progress.

---

## Requirements

- Python 3.10+
- PyTorch (pinned in `requirements.txt`)
- For the enclave: SGX-capable CPU and Gramine 1.5

## License

MIT — see [LICENSE](LICENSE).
