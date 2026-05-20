---
paths:
  - "mcy_dist_ai/**"
---

# mcy_dist_ai — Distributed Training Coordination

The PyPI-published package that drives synchronous data-parallel PyTorch training across nodes. Each process is either a `LEADER` (gradient aggregator) or a `WORKER` (batch trainer), selected by `--role` at startup. Nodes do not talk to each other directly — they read and write signal files in `BASE_DIR` (cwd captured at package import time), and an out-of-band transport layer (the Vulkan P2P network) ships those files between hosts. This package only assumes the files arrive.

## Design intent

- **File-based IPC, not sockets.** Gramine SGX enclaves restrict network APIs, so coordination uses files declared as enclave inputs/outputs. The transport layer is somebody else's problem.
- **Two-file ready-pattern.** Producer writes `<name>.pth`, then touches `<name>_ready.pth`. Consumer polls the ready file, then reads the data file, then deletes the ready file. The post-detection `asyncio.sleep(WAITING_PERIOD)` is a stand-in for proper locking — don't remove it without replacing the wait mechanism. (A separate, longer sleep guards the user-script transfer in `wait_file_transfer_complete`.)
- **Single-worker short-circuit.** With one worker, no leader runs (`args.py` exits cleanly) and the worker skips state-dict round-trips. Preserve this path; many users run single-worker.
- **`monitor.pth` is a liveness heartbeat.** Each role periodically touches it so the surrounding orchestrator can detect a stuck enclave. Don't repurpose it.

## Patterns

- Role dispatch lives in `main.py`; everything else is role code or shared helpers. New roles go through `ROLE` + `InvalidRole`, not through ad-hoc branches.
- Paths are derived from `constants.BASE_DIR` (always `os.getcwd()` at import time). Tests rely on `os.chdir` before importing `mcy_dist_ai.main` — never compute paths relative to `__file__`.
- Per-worker files use the `<stem>_<node>.<ext>` suffix scheme implemented in `Leader.get_path` / `tests.utils.leader_get_path`. Keep these two in sync.
- Async tasks come in pairs: a work coroutine plus a `monitor` coroutine, gathered together. New long-running work follows the same shape.

## Conventions

- Constants and filename templates live in `constants.py`. Don't inline literal `.pth` paths in role code.
- Logging goes through `mcy_dist_ai.logger.logger`. `LOG_INTERVAL` gates per-step info logs to keep training output readable.
- `WAITING_PERIOD` (poll interval) and `MONITORING_PERIOD` (heartbeat) are tuned for filesystem polling; treat as protocol constants, not knobs.

## Gotchas

- `args.py` calls `sys.exit` at import time on bad input or single-worker leader. Anything that imports `mcy_dist_ai.main` inherits that — tests patch `sys.argv` before import.
- The `signal_worker_finished` call must remain **after** `save_gradient` on the last iteration (see the TODO in `worker.py`); moving it above the gradient handoff breaks the Vulkan confirmation flow.
- `experimental/` and the `WORKER-LLM` role referenced in tests are unfinished — not part of the supported flow.
