---
paths:
  - "mcy_dist_ai/utils.py"
  - "mcy_dist_ai/import_user_files.py"
  - "mcy_dist_ai/script/**"
---

# User-Script Integration

Layers on `mcy-dist-ai.md`. This is the seam between the framework and user code: the framework imports `user_script.py` and `user_requirements.txt` shipped by the caller, installs the requirements, and calls a fixed set of symbols. Everything model- and data-specific lives in the user script; the framework stays generic.

## Design intent

- **Contract over inheritance.** The user script is a plain module, not a subclass — it just has to expose the required symbols. Treat the symbol list as a stable API; renaming any of them is a breaking change for every downstream user.
- **Dynamic install at startup.** `wait_and_install_user_requirements` pip-installs `user_requirements.txt` into the running interpreter before `utils.py` finishes importing. Side-effecting imports at module top level is intentional — the user script and its deps must be ready before any role code runs.
- **Two data-loader paths.** `--tensor_load` reads pre-split `data_tensor.pt`/`target_tensor.pt` produced by `mcy-split-data`; otherwise the user's `create_data_loader` runs inside the worker. The CLI exists so heavy preprocessing happens once, not per worker.

## Required user_script.py surface

- `N_EPOCHS: int`, `BATCH_SIZE: int`
- `create_model() -> nn.Module`
- `create_optimizer(model) -> Optimizer`
- `create_data_loader(path) -> DataLoader`
- `create_extra_training_args(data_loader, optimizer)` — may return `None`, a single object, or an iterable; `safe_create_extra_training_args` normalises to a list
- `train_batch(batch, model, optimizer, *extra_args)` — must call `optimizer.zero_grad()` itself (the framework does not), then `loss.backward()`, return the loss for logging

Full prose contract lives in `docs/user_script_requirements.md` — keep that doc and this rule aligned when the surface changes.

## Patterns

- `torch_safe_load` retries `torch.load` because the file may still be mid-copy from the transport layer. Use it for any tensor read whose producer is a different process.
- Checkpoint format is `struct.pack('!ii', epoch, batch_idx + 1)` — network-byte-order ints, "next batch to run". Don't change without bumping a version byte; old checkpoint files would misparse silently.
- The split CLI (`mcy-split-data`) loads the entire dataset into memory (see TODO in `script/split_data.py`). Don't add features that assume streaming.

## Gotchas

- `wait_file_transfer_complete` sleeps 2s after the file appears — same locking workaround as the rest of the package. Don't remove it.
- `user_script` is imported as a module-level global in `utils.py`. Anything imported from `mcy_dist_ai.utils` triggers the wait-install-import sequence, including in tests.
