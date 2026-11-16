# echowarp/vllm `publish-20261113-xpu`, 2026-11-16: first public release

For Intel Arc Pro B70 (XPU), on `llmt/v0.26-xpu` (v0.26.0 + Intel's llm-scaler patch + the same feature commits + 3 XPU-only). What each commit does: [README.md](README.md).

## What is in it

- The `llmt/v0.26-xpu` feature commits (one feature each: spec decode, Mamba/GDN state handling, KV offload for hybrid
  models, scheduler liveness fixes, prefix-cache tail hold).
- Packed (cross-layer) KV layout for hybrid attention + Mamba/GDN models (`"enable_cross_layers_blocks": "true"`).
- Named extension hooks for memory-management plugins (`vllm/v1/extension_hooks.py`) and their call sites; lent
  blocks left out of KV usage.
- EXL3_* switch defaults equal the reference build only while the EXL3 plugin serves the model; stock behaviour
  otherwise.
- Fixes: a fresh pinned copy of the drafter's backup next tokens per step (race), and the KV-NANCHECK scanner no
  longer reads back a failed store job.

## How it was validated

- Arc B70, production image with this branch's files mounted (2026-11-16): native hooks (logged), KV 497,302 tokens,
  LoRA decode 179.9 tok/s (1 stream) / 161.2 (2 streams) with AllBroker's lend manager, 6 x 60k stress 0 OOM,
  minimum free device memory 557 MiB; log-probs on 6 prompts: 6/6 identical vs the same engine with the internal
  (pre-publish) plugin, 5/6 vs one without the plugin (a known plugin-load change).
- CPU: `py_compile` of the changed files. The new unit tests need a device and vLLM's conftest fixtures and were not
  run on CPU. AllBroker's plugin CPU suite passes with this branch mounted (native hooks).

## Known limitations

- Not yet run from a freshly built image (AllRunner's Dockerfiles are rebuilt from the published commits next).
- vLLM logs `MAMBA-ACC` warnings under load: a request's accepted-token count inherited across a batch condense is
  reset to 1 (logged for the first 20 and then every 1,000th correction). Harmless; the race itself is worked around,
  not removed.
- The hooks `model_runner.execute_model`, `model_forward` and `propose_draft` exist but nothing registers on them
  in this release.

