# vllm

A fork of [vLLM](https://github.com/vllm-project/vllm) v0.26.0 for serving Qwen3.8-27B (EXL3 4.0 bpw, a hybrid
attention + GDN model) to several tool-calling agents at once on one RTX 5090. It adds small, one-feature-per-commit
changes to the scheduler, speculative decoding, the prefix cache and KV offload for hybrid models, and named extension
hooks for a memory-management plugin. With the [exl3cuda](https://github.com/echowarp/exllamav3) kernels and
[AllBroker](https://github.com/echowarp/AllBroker), on shipped defaults: 1,262.5 tok/s aggregate at 16 concurrent
requests (vLLM 0.28: 546.6; SparkInfer v0.5.5: 345.1) and 337.6 tok/s single stream (SparkInfer + DSpark v2: 264.8).

This default branch is the NVIDIA build (tested on the RTX 5090, sm_120). The Intel Arc Pro B70 build is on branch
[`xpu`](https://github.com/echowarp/vllm/tree/xpu).

## What's different from upstream

Every commit and hook, the `EXL3_*` switch defaults and the CPU tests: [docs/llmt/README.md](docs/llmt/README.md).
Release notes and validation: [docs/llmt/RELEASE.md](docs/llmt/RELEASE.md).

| Change | What it does | Measured effect |
|---|---|---|
| Prefix-cache tail hold | keeps a finished request's partial last block and its GDN state at the exact token, so the next agent turn (a tool result) continues where the last one stopped | real RTX 5090 agent traffic: 232 tail hits reused 635,441 tokens beyond the block-rounded prefix hit, median 2,085 per hit (1) |
| Packed (cross-layer) KV for hybrid models | one KV allocation, one contiguous row per block across all layers, for attention + Mamba/GDN models | RTX 5090 KV cache 436,585 -> 482,846 tokens (+10.6 %), together with AllBroker's lazy vision tower and vision-tower lending (2) |
| KV offload for hybrid models | CPU-tier offload that works with GDN state and MTP; per-slot pinning for a broker-managed shared pool; paced copies; NaN/inf scan of stored blocks | the host-RAM tier served 2.6 % of prompt tokens on real RTX 5090 agent traffic, next to 59.7 % from the GPU prefix cache (1) |
| MTP speculative decoding | full prefix hit kept with MTP drafters, draft depth keyed on decoding requests, stop-token truncation on the GPU, fused draft sampling, FlashInfer top-k/top-p, faster GQA verify attention | not measured per commit; see the whole-stack numbers above |
| Host work off the critical path | the CPU prepares step n+1 while the GPU runs step n: no wait for the accepted-token counts on spec-decode steps in Mamba align mode (also on steps that add, finish or reorder requests); GDN recompute metadata through a ring of pinned buffers instead of synchronous pageable copies; stop-token truncation and draft sampling on the GPU | each of the two sync removals took out ~2.6 ms of GPU idle per decode step (of 18.4 ms) on an RTX 5090 |
| Scheduler liveness | no stall on blocks held by waiting requests, length-capped finish, liveness guard, recompute instead of abort on NaN/inf logits | correctness, no speed claim |
| Extension hooks | 15 named hooks (scheduler, block pool, model runner, worker) instead of runtime method patching; lent blocks left out of KV usage | log-probs vs the production engine: 5/6 prompts identical, max abs difference 0.070, inside the 0.091 between two production starts (3) |
| `EXL3_*` switch defaults | the reference build's values apply only while the EXL3 plugin serves an EXL3 model on sm_120 | every other model keeps upstream behaviour |

Sources: (1) AllRunner [benchmarks/charts/data/ours.json](https://github.com/echowarp/AllRunner/blob/main/benchmarks/charts/data/ours.json),
block `cache_production_5090`; (2) AllRunner chart [on-demand](https://github.com/echowarp/AllRunner/blob/main/benchmarks/charts/out/on-demand.png)
and [docs/llmt/RELEASE.md](docs/llmt/RELEASE.md); (3) [docs/llmt/RELEASE.md](docs/llmt/RELEASE.md); whole-stack
numbers: AllRunner [benchmarks/README.md](https://github.com/echowarp/AllRunner/blob/main/benchmarks/README.md).


The tail hold and the RAM tier together: what an agent's next turn has to prefill.

![Prefill only what's new](https://github.com/echowarp/AllRunner/raw/main/benchmarks/charts/out/tail-reuse.png)

## Use it

The commits change Python files only. [AllRunner](https://github.com/echowarp/AllRunner)'s RTX 5090 image starts from
the stock `vllm/vllm-openai:v0.26.0` image and copies this branch's changed files over the installed vLLM
([docker/overlay_vllm.sh](https://github.com/echowarp/AllRunner/blob/main/docker/overlay_vllm.sh) checks that every
file it replaces is the v0.26.0 original), then adds exl3cuda and AllBroker's `llmt_kv`:

```sh
git clone https://github.com/echowarp/AllRunner && cd AllRunner
docker build -f docker/Dockerfile.rtx5090 -t allrunner-rtx5090 .
launchers/rtx5090.sh        # engine on 127.0.0.1:8001
```

Launcher settings, start order and the router in front: AllRunner
[launchers/README.md](https://github.com/echowarp/AllRunner/blob/main/launchers/README.md). A full source build of
this tree follows upstream's build instructions; it is not what we test.

## Requires / works with

- [echowarp/exllamav3](https://github.com/echowarp/exllamav3): the `exl3cuda` plugin with the EXL3 and attention
  kernels for the RTX 5090.
- [echowarp/AllBroker](https://github.com/echowarp/AllBroker): the memory layer (host-RAM KV tier, weight keeper, KV
  lending). Its plugin works with stock vLLM 0.26 and uses this fork's extension hooks when they are there.
- [echowarp/AllRunner](https://github.com/echowarp/AllRunner): images, launchers, router and benchmarks.
- Arc Pro B70: branch [`xpu`](https://github.com/echowarp/vllm/tree/xpu) with
  [echowarp/exl3xpu](https://github.com/echowarp/exl3xpu).

## Upstream

vLLM is by the vLLM project: [vllm-project/vllm](https://github.com/vllm-project/vllm). This fork is based on tag
v0.26.0; for everything else (other models and hardware, installation, documentation) see upstream's
[README](https://github.com/vllm-project/vllm/blob/v0.26.0/README.md) and [docs.vllm.ai](https://docs.vllm.ai).

## License

Apache-2.0, as upstream ([LICENSE](LICENSE)).
