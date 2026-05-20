# mcy-sgx-gramine

## Rule Files — Progressive Disclosure

Rules in `.claude/rules/` contain curated architectural knowledge: design intent, patterns, constraints, and conventions.

**Before answering questions, researching, or modifying code in any area below:** read the relevant rule file(s) FIRST. They explain the "why" and "how" without needing to scan source files. Only dive into source code for details the rules don't cover.

Loading is Claude-driven, not harness-driven: no hook auto-reads these files when an Edit/Write matches the paths in the table. The `paths:` frontmatter on each rule file is documentation-only.

| Rule file | Applies to | Covers |
|---|---|---|
| `mcy-dist-ai.md` | `mcy_dist_ai/**` | Package-wide design: LEADER/WORKER roles, file-signal IPC under `BASE_DIR`, single-worker short-circuit, async monitor pattern. |
| `mcy-dist-ai-user-script.md` | `mcy_dist_ai/utils.py`, `mcy_dist_ai/import_user_files.py`, `mcy_dist_ai/script/**` | Layers on `mcy-dist-ai.md`. User-script symbol contract, dynamic install/import flow, tensor-load vs user-loader paths, checkpoint format. |
| `sgx-enclave-app.md` | `app/**` | Gramine SGX enclave: attestation gate, ECDH+Fernet channels for data and model, manifest trust boundaries, enclave vs host entrypoints. |
| `sgx-remote-client.md` | `remote/**` | Untrusted counterpart to `app/`: protocol mirror, local baseline round-trip, why utils is duplicated across the trust boundary. |
| `tests.md` | `tests/**` | Multi-process simulation harness (`WatchLeader`/`WatchWorker` replace P2P transport), `with_temp_dir` lifecycle, examples directory layout. |
