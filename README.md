# echowarp/vllm `publish-20261113-cuda`: vLLM 0.26 for Qwen3.8-27B agent serving on an RTX 5090

This fork is vLLM v0.26.0 plus small, one-feature-per-commit changes (scheduler, speculative decoding, prefix cache and
KV offload for hybrid attention + GDN models) and named extension hooks for a memory-management plugin. Fork
documentation: [docs/llmt/README.md](docs/llmt/README.md) (every commit, the hooks, the `EXL3_*` switches, CPU tests);
release notes: [docs/llmt/RELEASE.md](docs/llmt/RELEASE.md).

## What this fork adds

| Feature | What it does | Measured effect |
|---|---|---|
| Prefix-cache tail hold | keeps a finished request's partial last block and its GDN state at the exact token, so the next agent turn (a tool result) continues where the last one stopped | real RTX 5090 agent traffic: 232 tail hits reused 635,441 tokens beyond the block-rounded prefix hit, median 2,085 per hit (1) |
| Packed (cross-layer) KV for hybrid models | one KV allocation, one contiguous row per block across all layers, for attention + Mamba/GDN models | RTX 5090 KV cache 436,585 -> 482,846 tokens (+10.6 %), together with AllBroker's lazy vision tower and vision-tower lending (2) |
| KV offload for hybrid models | CPU-tier offload that works with GDN state and MTP; per-slot pinning for a broker-managed shared pool; paced copies; NaN/inf scan of stored blocks | the host-RAM tier served 2.6 % of prompt tokens on real RTX 5090 agent traffic, next to 59.7 % from the GPU prefix cache (1) |
| MTP speculative decoding | full prefix hit kept with MTP drafters, draft depth keyed on decoding requests, stop-token truncation on the GPU, fused draft sampling, FlashInfer top-k/top-p, faster GQA verify attention | not measured per commit; see the whole-stack numbers below |
| Scheduler liveness | no stall on blocks held by waiting requests, length-capped finish, liveness guard, recompute instead of abort on NaN/inf logits | correctness, no speed claim |
| Extension hooks | 15 named hooks (scheduler, block pool, model runner, worker) instead of runtime method patching; lent blocks left out of KV usage | log-probs vs the production engine: 5/6 prompts identical, max abs difference 0.070, inside the 0.091 between two production starts (3) |
| `EXL3_*` switch defaults | the reference build's values apply only while the EXL3 plugin serves an EXL3 model on sm_120 | every other model keeps upstream behaviour |

Whole stack (this branch + exl3cuda kernels + AllBroker, RTX 5090, shipped defaults): 1,262.5 tok/s aggregate at 16
concurrent requests (vLLM 0.28: 546.6; SparkInfer v0.5.5: 345.1) and 337.6 tok/s single stream, overall harmonic mean
(SparkInfer + DSpark v2: 264.8) (4).

Sources: (1) AllRunner [benchmarks/charts/data/ours.json](https://github.com/echowarp/AllRunner/blob/main/benchmarks/charts/data/ours.json),
block `cache_production_5090`; (2) AllRunner chart [on-demand](https://github.com/echowarp/AllRunner/blob/main/benchmarks/charts/out/on-demand.png) and [docs/llmt/RELEASE.md](docs/llmt/RELEASE.md); (3) [docs/llmt/RELEASE.md](docs/llmt/RELEASE.md);
(4) AllRunner [benchmarks/README.md](https://github.com/echowarp/AllRunner/blob/main/benchmarks/README.md).

## Which branch

- `publish-20261113-cuda` (this branch): NVIDIA, tested on the RTX 5090 (sm_120).
- `publish-20261113-xpu`: Intel Arc Pro B70; Intel's llm-scaler XPU patch is its first commit.
- `main` is upstream vLLM, unchanged.

## How it fits

[AllRunner](https://github.com/echowarp/AllRunner) builds one image per GPU from this branch, the GPU's kernel plugin
([echowarp/exllamav3](https://github.com/echowarp/exllamav3) `exl3cuda` here) and AllBroker's `llmt_kv`, pinned by
commit, and runs the router, launchers and benchmarks. [AllBroker](https://github.com/echowarp/AllBroker) is the memory
layer (host-RAM KV tier, weight keeper, KV lending); its plugin works with stock vLLM 0.26 and uses this branch's
extension hooks when they are there.

## License

Apache-2.0, as upstream ([LICENSE](LICENSE)). Upstream's README follows unchanged.

---

## Upstream README

<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
