# echowarp/vllm `publish-20261113-cuda`, 2026-11-16: first public release

For NVIDIA RTX 5090 (sm_120), on `llmt/v0.26-cuda` (v0.26.0 + 34 feature commits). What each commit does: [README.md](README.md).

## What is in it

- The `llmt/v0.26-cuda` feature commits (one feature each: spec decode, Mamba/GDN state handling, KV offload for hybrid
  models, scheduler liveness fixes, prefix-cache tail hold).
- Packed (cross-layer) KV layout for hybrid attention + Mamba/GDN models (`"enable_cross_layers_blocks": "true"`).
- Named extension hooks for memory-management plugins (`vllm/v1/extension_hooks.py`) and their call sites; lent
  blocks left out of KV usage.
- EXL3_* switch defaults equal the reference build only while the EXL3 plugin serves the model; stock behaviour
  otherwise.
- Fixes: a fresh pinned copy of the drafter's backup next tokens per step (race), and the KV-NANCHECK scanner no
  longer reads back a failed store job.

## How it was validated

- RTX 5090, production image with the 21 files this branch changes mounted (2026-11-16): native hooks (logged), KV
  482,846 tokens with the packed layout, vision tower lent and returned, agent simulation 79-82 turns with 0 errors,
  6 x 60k-token + images stress with 0 OOM / tracebacks / EngineDead; log-probs vs the production engine 5/6 prompts
  identical, max |dlp| 0.070, inside the start-to-start noise floor (0.091 between two production starts).
- CPU: `py_compile` of the changed files. The new unit tests need a device and vLLM's conftest fixtures and were not
  run on CPU. AllBroker's plugin CPU suite passes with this branch mounted (native hooks).

## Known limitations

- Not yet run from a freshly built image (AllRunner's Dockerfiles are rebuilt from the published commits next).
- vLLM logs `MAMBA-ACC` warnings under load: a request's accepted-token count inherited across a batch condense is
  reset to 1 (logged for the first 20 and then every 1,000th correction). Harmless; the race itself is worked around,
  not removed.
- The hooks `model_runner.execute_model`, `model_forward` and `propose_draft` exist but nothing registers on them
  in this release.
- Two starts of the same engine are not bit-identical (4/6 greedy prompts identical, max |dlp| 0.091 on the 5090).
- The paced KV staging ring for RAM-tier loads is on the XPU branch only.
