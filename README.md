# vllm (Intel Arc Pro B70 build)

A fork of [vLLM](https://github.com/vllm-project/vllm) v0.26.0 for serving Qwen3.8-27B (EXL3 4.0 bpw, a hybrid
attention + GDN model) to several tool-calling agents at once on one Intel Arc Pro B70. Its first commit is Intel's
[llm-scaler](https://github.com/intel/llm-scaler) XPU patch for vLLM 0.26.0; on top come the same one-feature-per-commit
changes as the NVIDIA build (scheduler, speculative decoding, prefix cache and KV offload for hybrid models, extension
hooks for a memory-management plugin) and three XPU-only commits. With the
[exl3xpu](https://github.com/echowarp/exl3xpu) kernels and [AllBroker](https://github.com/echowarp/AllBroker), on
upstream exl3xpu's own benchmark: aggregate decode 125.0 tok/s at 1 stream and 530.2 at 16 (prose; upstream exl3xpu
91.2 and 365), cold prefill of a 32K-token prompt 2,823 tok/s (upstream 2,124).

This branch (`xpu`) is the Intel build; the NVIDIA build is the repository's
[default branch](https://github.com/echowarp/vllm).

## What's different from upstream

Every commit and hook, the `EXL3_*` switch defaults and the CPU tests: [docs/llmt/README.md](docs/llmt/README.md).
Release notes and validation: [docs/llmt/RELEASE.md](docs/llmt/RELEASE.md).

| Change | What it does | Measured effect |
|---|---|---|
| Prefix-cache tail hold | keeps a finished request's partial last block and its GDN state at the exact token, so the next agent turn (a tool result) continues where the last one stopped | same commit as the NVIDIA build; on real RTX 5090 agent traffic 232 tail hits reused 635,441 tokens beyond the block-rounded prefix hit, median 2,085 per hit (1) |
| Packed (cross-layer) KV for hybrid models | one KV allocation, one contiguous row per block across all layers, for attention + Mamba/GDN models | measured on the RTX 5090 (NVIDIA build); not measured separately on the Arc |
| KV offload for hybrid models | CPU-tier offload that works with GDN state and MTP; paced copies through a pinned staging ring (XPU only); NaN/inf scan of stored blocks | the host-RAM tier served 2.6 % of prompt tokens on real RTX 5090 agent traffic (1); not measured separately on the Arc |
| MTP speculative decoding | full prefix hit kept with MTP drafters, draft depth keyed on decoding requests, stop-token truncation on the GPU, fused draft sampling, fused top-k/top-p mask from exl3xpu (XPU only) | not measured per commit; see the whole-stack numbers above |
| Host work off the critical path | the CPU prepares step n+1 while the GPU runs step n: no wait for the accepted-token counts on spec-decode steps in Mamba align mode (also on steps that add, finish or reorder requests); GDN recompute metadata through a ring of pinned buffers instead of synchronous pageable copies; stop-token truncation and draft sampling on the GPU | each of the two sync removals took out ~2.6 ms of GPU idle per decode step (of 18.4 ms) on an RTX 5090 |
| Scheduler liveness | no stall on blocks held by waiting requests, length-capped finish, liveness guard, recompute instead of abort on NaN/inf logits | correctness, no speed claim |
| Extension hooks | 15 named hooks (scheduler, block pool, model runner, worker) instead of runtime method patching; lent blocks left out of KV usage | Arc B70 with AllBroker's lend manager: KV 497,302 tokens, LoRA decode 179.9 tok/s (1 stream); log-probs 6/6 prompts identical vs the same engine with the pre-publish plugin (2) |
| `EXL3_*` switch defaults | the reference build's values apply only while the exl3xpu plugin serves an EXL3 model | every other model keeps upstream behaviour |

Sources: (1) AllRunner [benchmarks/charts/data/ours.json](https://github.com/echowarp/AllRunner/blob/main/benchmarks/charts/data/ours.json),
block `cache_production_5090`; (2) [docs/llmt/RELEASE.md](docs/llmt/RELEASE.md); whole-stack numbers: AllRunner
[benchmarks/README.md](https://github.com/echowarp/AllRunner/blob/main/benchmarks/README.md).


The tail hold and the RAM tier together: what an agent's next turn has to prefill.

![Prefill only what's new](https://github.com/echowarp/AllRunner/raw/main/benchmarks/charts/out/tail-reuse.png)

## Use it

The commits after Intel's patch change Python files only. [AllRunner](https://github.com/echowarp/AllRunner)'s Arc
image starts from an `intel/llm-scaler-vllm:0.26.0-b2`-based image and copies this branch's changed files over Intel's
installed vLLM ([docker/overlay_vllm.sh](https://github.com/echowarp/AllRunner/blob/main/docker/overlay_vllm.sh)
checks that every file it replaces equals the Intel files the branch was made from), then adds exl3xpu and
AllBroker's `llmt_kv`:

```sh
git clone https://github.com/echowarp/AllRunner && cd AllRunner
docker build -f docker/Dockerfile.arc -t allrunner-arc .
launchers/arc-b70.sh        # engine on 127.0.0.1:8000
```

Launcher settings, start order and the router in front: AllRunner
[launchers/README.md](https://github.com/echowarp/AllRunner/blob/main/launchers/README.md). A full source build of
this tree is not what we test.

## Requires / works with

- [echowarp/exl3xpu](https://github.com/echowarp/exl3xpu): the ESIMD kernels and vLLM plugin for the Arc Pro B70.
- [echowarp/AllBroker](https://github.com/echowarp/AllBroker): the memory layer (host-RAM KV tier, weight keeper, KV
  lending). Its plugin works with stock vLLM 0.26 and uses this fork's extension hooks when they are there.
- [echowarp/AllRunner](https://github.com/echowarp/AllRunner): images, launchers, router and benchmarks.
- [echowarp/b70-linux](https://github.com/echowarp/b70-linux): optional host patches and tools for the Arc Pro B70.
- RTX 5090: the [default branch](https://github.com/echowarp/vllm) with
  [echowarp/exllamav3](https://github.com/echowarp/exllamav3) (`exl3cuda`).

## Upstream

vLLM is by the vLLM project: [vllm-project/vllm](https://github.com/vllm-project/vllm). This fork is based on tag
v0.26.0 plus Intel's [llm-scaler](https://github.com/intel/llm-scaler) `vllm-0.26.0-b2` patch (the first commit, written by
Intel and imported unchanged). For everything else (other models and hardware, installation, documentation) see upstream's
[README](https://github.com/vllm-project/vllm/blob/v0.26.0/README.md) and [docs.vllm.ai](https://docs.vllm.ai).

## License

Apache-2.0, as upstream ([LICENSE](LICENSE)); Intel's patch is Apache-2.0 as well.
