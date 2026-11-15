# echowarp/vllm `publish-20261113-*`: what this branch adds over vLLM v0.26.0

Two branches share one design:

| Branch | Base | For |
|---|---|---|
| `publish-20261113-cuda` | `llmt/v0.26-cuda` (v0.26.0 + 34 feature commits) | NVIDIA (tested on RTX 5090, sm_120) |
| `publish-20261113-xpu` | `llmt/v0.26-xpu` (v0.26.0 + Intel's llm-scaler patch + the same feature commits + 3 XPU-only) | Intel Arc Pro B70 (XPU) |

The `llmt/v0.26-*` commits are one feature per commit, each with its own message (spec decode, Mamba/GDN state
handling, KV offload for hybrid models, scheduler liveness fixes, prefix-cache tail hold). `git log v0.26.0..HEAD`
lists them. This branch adds five more, all small and meant for upstream:

1. `[Bugfix][Hybrid][KV Cache] Mamba/GDN state views in the cross-layer (packed) KV layout`
2. `[Bugfix][Hybrid][KV Cache] KVBlockZeroer: separate block stride from page size`
3. `[Bugfix][KV Offload][Hybrid] Per-slot views when registering a packed hybrid KV cache`
4. `[Core][Plugins] Named extension hooks for memory-management plugins`
5. `[Core][Plugins] Extension hook call sites; lent blocks left out of KV usage`

Commits 1-3 make `"enable_cross_layers_blocks": "true"` (in `kv_connector_extra_config`) work for hybrid
attention + Mamba/GDN models such as Qwen3-Next. With it the KV pool is one allocation in which one block is one
contiguous row holding every layer's page. Nothing changes in the default (unpacked) layout; the tests check that.

Commits 4-5 let a memory-management plugin (for example AllBroker's `llmt_kv`) attach to the scheduler, block pool,
model runner and worker through named hooks instead of replacing methods at runtime.

## Extension hooks (`vllm/v1/extension_hooks.py`)

```python
from vllm.v1 import extension_hooks as hooks

def register():                     # a vllm.general_plugins entry point: runs in every vLLM process
    hooks.register("scheduler.before_schedule", on_step)
    hooks.register("block_pool.get_new_blocks", place_blocks)
```

| Name | Kind | Called | Arguments -> result |
|---|---|---|---|
| `scheduler.init` | notify | end of `Scheduler.__init__` | `(scheduler)` |
| `scheduler.before_schedule` | notify | start of `Scheduler.schedule` | `(scheduler)` |
| `scheduler.after_schedule` | notify | end of `Scheduler.schedule` | `(scheduler, scheduler_output)` |
| `scheduler.admit` | first | for each waiting request, before it is considered | `(scheduler, request)` -> `False` keeps it waiting this step (it goes to `skipped_waiting`); `None`/`True` = default |
| `block_pool.get_new_blocks` | first | start of `BlockPool.get_new_blocks` (after the free-count check) | `(pool, num_blocks)` -> exactly `num_blocks` blocks taken from the free queue (ref_cnt incremented, cached hash evicted), or `None` = default |
| `block_pool.cache_full_blocks` | notify | end of `BlockPool.cache_full_blocks` (only when new blocks were cached) | `(pool, request, num_cached_blocks, num_full_blocks)` |
| `block_pool.touch` | notify | start of `BlockPool.touch` | `(pool, blocks)` |
| `block_pool.get_cached_block` | notify | start of `BlockPool.get_cached_block` | `(pool, block_hash)` |
| `kv_cache_manager.allocate_slots` | context | around `KVCacheManager.allocate_slots` | `(manager, request)` |
| `model_runner.kv_caches_initialized` | notify | end of `GPUModelRunner.initialize_kv_cache_tensors` | `(runner, kv_caches, kv_cache_config)` |
| `model_runner.execute_model` | context | around `GPUModelRunner.execute_model` | `(runner)` |
| `model_runner.model_forward` | context | around `GPUModelRunner._model_forward` | `(runner, args, kwargs)` |
| `model_runner.propose_draft` | context | around `GPUModelRunner.propose_draft_token_ids` | `(runner)` |
| `worker.available_kv_memory` | fold | result of `Worker.determine_available_memory` | `(bytes, worker)` -> bytes |
| `worker.warmed_up` | notify | end of `Worker.compile_or_warm_up_model` | `(worker)` |

Kinds: **notify** runs every callable and logs (does not raise) their exceptions; **first** returns the first
non-`None` result; **fold** chains `value = fn(value, *args)`; **context** runs the call inside the context managers
the callables return. Unknown names raise `ValueError` at registration. With nothing registered a call site is one
dict lookup. The XPU model runner and worker inherit these methods, so the same hooks fire on both branches.

`BlockPool.num_lent_blocks` (default 0) is the one plain state change: a plugin that takes blocks out of the pool
for another use sets it, and `get_usage()`, the scheduler's length-cap wait and the tail-hold watermark leave those
blocks out.

## EXL3_* switch defaults (`vllm/v1/llmt_defaults.py`)

The branch's engine features are switched by `EXL3_*` variables. A variable that is set (also to an empty string)
always wins. An unset one takes the reference build's value **only while the EXL3 plugin serves the model**: the
model's quantization is `exl3` (a method that only exl3cuda registers) and the GPU is sm_120 (RTX 5090). Every other model, and
this branch without the plugin, keeps upstream behaviour. `activate(vllm_config)` decides once per process (first
thing in `Scheduler.__init__` and `GPUModelRunner.__init__`); module constants made with `flag()` are re-evaluated
then, `env()` reads the decision on every call.

| Variable | Upstream (inactive) | While the plugin serves an EXL3 model |
|---|---|---|
| `EXL3_ALIGN_NOSYNC` | 0 | 1 |
| `EXL3_SPEC_STOP_TRUNC` | 0 | 1 |
| `EXL3_TAIL_HOLD` | 0 | 1 (inert until the runner confirms GDN recompute) |
| `EXL3_TAIL_HOLD_FREE` | 0.15 | 0.03 |
| `EXL3_EAGLE_KEEP` | 0 | 2 (core and offloading-connector scheduler) |
| `EXL3_SCHED_DECODE_STEPS` | 0 | 1 |
| `EXL3_DYN_SD_DECODE_COUNT` | 0 | 1 |
| `EXL3_ENCODER_CACHE_TOKENS` | 0 | 16384 |
| `EXL3_GPU_EVICT_ORDER` | (vLLM order) | chain |
| `EXL3_FI_TOPKP` | 0 | 1 (rejection sampler and top-k/top-p sampler) |
| `EXL3_ATTN_SEGMENTS` | 32 | 64 |
| `EXL3_ATTN_SEGMENTS_MANY` | 0 | 32 |
| `EXL3_ATTN_VERIFY_BM` | 0 | 32 |
| `EXL3_ATTN_VERIFY_WARPS` | 8 | 2 |
| `EXL3_GDN_RECOMPUTE` | off | follows the plugin's `GDN_RC["on"]` (on by default for EXL3 models; the plugin clears it when its half cannot install). `EXL3_GDN_RECOMPUTE=1` forces it for any model |
| `EXL3_RC_PINNED` | 1 | 1 (only used with GDN recompute) |

## Tests (CPU, no GPU needed)

```bash
TRITON_INTERPRET=1 VLLM_KV_CACHE_LAYOUT=NHD pytest -q --noconftest \
    tests/v1/core/test_extension_hooks.py tests/v1/worker/test_packed_kv_hybrid.py \
    tests/v1/test_llmt_defaults.py
```

`test_extension_hooks.py`: registry semantics, the decorators, block-pool placement override and lent-block usage,
the scheduler hooks with an admission hook holding one request back (uses a local tiny OPT config, no download).
`test_packed_kv_hybrid.py`: fake hybrid configs planned by vLLM's own planner, packed and unpacked: views read back
without aliasing, the zeroer (Triton interpreter) touches only the new blocks, offload registration gives the same
tensors and refs in both layouts, and a Qwen3-Next-sized geometry on meta tensors.

## Upstreaming

Each commit applies on its own to current upstream with small context fixes:

- Commits 1-3 are independent bug fixes (cut `pr/packed-kv-hybrid` from upstream main and cherry-pick them with the
  packed-KV test).
- Commits 4-5 are one RFC-sized change (`pr/extension-hooks`). Upstream has no `tail-hold` or length-cap wait, so
  drop those two `num_lent_blocks` lines there; the rest has no dependency on this branch.

License: Apache-2.0, as upstream (`LICENSE`).
