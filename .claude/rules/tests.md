---
paths:
  - "tests/**"
---

# Tests & Simulation Harness

End-to-end tests for `mcy_dist_ai` run the real `Leader`/`Worker` code in separate processes and replace the Vulkan P2P transport with a local harness that shuttles files between per-process working directories under `tests/temp/`.

## Design intent

- **No mocks of the package under test.** `run_node` patches `sys.argv` and calls the real `mcy_dist_ai.main.main()` in a child process. Tests assert on the trained model's accuracy, not on internal calls.
- **`WatchLeader` / `WatchWorker` *are* the transport layer.** They poll the worker's `gradient_ready.pth`, copy gradients into the leader's `<file>_<node>.<ext>` slot, copy aggregated state dicts back, and signal `worker_finished`. Read these two classes to understand the protocol contract from the outside in.
- **Examples are self-contained.** Each `tests/examples/<name>/` directory ships its own `user_script.py`, `user_requirements.txt`, `preprocess_data.py`, and `checks.py`. New examples follow the same shape so the harness can find each file by name.

## Patterns

- `@with_temp_dir` wraps tests that need a clean `tests/temp/`. `clear_tmp_dir_end=False` is used when you want to inspect artifacts after a failure — keep this opt-out.
- Parallel tests spawn `multiprocessing.Process` per worker + one for the leader + one for `simulate_p2p_network`. Sequential tests run workers one-by-one and pass the trained model from each worker as the starting state of the next.
- `dynamic_import` is the test-side equivalent of the framework's user-script loader. Use it (not `importlib` directly) so example modules behave consistently across tests.

## Conventions

- Filename and waiting-period constants are duplicated here (`tests/constants.py`) rather than imported from `mcy_dist_ai.constants` so the test harness can evolve without dragging the package along. When the package's filenames change, update both files.
- Test logging goes through `tests.logger.testlogger`, not the package logger, so harness output is distinguishable from node output in interleaved logs.

## Gotchas

- `run_node` calls `os.chdir` and imports `mcy_dist_ai.main` inside the child — that import has side effects (argparse, user-script install). Don't import the package at the top of a test module.
- The `WORKER-LLM` role used by `test_one_worker_llm` is not implemented in `mcy_dist_ai.main`; that test exercises the experimental path and is expected to be incomplete.
- `tests/unit/test_dhke/` contains `.go` and `.py` reference implementations of the key exchange, not pytest cases. Don't add them to a test collection.
