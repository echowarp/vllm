# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (
    THRESHOLD_BYTES,
    swap_blocks_batch,
)

logger = init_logger(__name__)


# ---- RAM-tier store NaN scan -------------------------------------------------------------------------
# Every GPU->CPU store is scanned for NaN/inf in its RAM copy by one background thread (nice 10) before
# the store is reported complete; a store with a non-finite value is reported failed, so the scheduler
# frees those chunks instead of making them available to later prefix hits ("OFFLOAD-NAN:" lines).
# Page layouts per KV group / data ref come from the connector worker (set_layout). EXL3_KV_NANCHECK=0
# disables.
import ctypes as _ct
import os as _os
import queue as _queue
import threading as _threading

_NC_MASKS = {  # dtype -> (numpy uint view, abs mask, first non-finite pattern)
    torch.float16: (np.uint16, 0x7FFF, 0x7C00),
    torch.bfloat16: (np.uint16, 0x7FFF, 0x7F80),
    torch.float32: (np.uint32, 0x7FFFFFFF, 0x7F800000),
}
_NC_LAYOUT: list[list[list[tuple[int, int, torch.dtype]]]] | None = None


def set_layout(group_segs) -> None:
    """group -> data ref -> [(byte offset in page, bytes, dtype)]."""
    global _NC_LAYOUT
    _NC_LAYOUT = group_segs


class _NanChecker:
    def __init__(self):
        self.on = _os.environ.get("EXL3_KV_NANCHECK", "1") == "1"
        self.q: _queue.Queue = _queue.Queue()
        self.done: _queue.Queue = _queue.Queue()
        self.n_jobs = self.n_bad = self.n_bytes = 0
        self.t_scan = 0.0
        self.last_log = time.monotonic()
        self.buf = np.empty(0, dtype=np.uint8)
        self.th = None
        if self.on:
            self.th = _threading.Thread(target=self._run, name="kv-nancheck", daemon=True)
            self.th.start()
            logger.info("KV-NANCHECK on: RAM-tier stores are scanned for NaN/inf before they become hits")

    def _run(self):
        try:
            _os.setpriority(_os.PRIO_PROCESS, _threading.get_native_id(), 10)
        except Exception:
            pass
        while True:
            item = self.q.get()
            if item is None:
                return
            result, ops = item
            t0 = time.monotonic()
            bad = None
            try:
                # never read back a failed store: its pool pages may be unallocated (full or over-quota tmpfs), and a read fault
                # there SIGBUSes the engine
                if result.success:
                    bad = self._scan(result.job_id, ops)
            except Exception as e:  # never fail the engine on a scanner bug
                logger.warning("KV-NANCHECK: scan error on store job %d: %r", result.job_id, e)
            dt = time.monotonic() - t0
            self.t_scan += dt
            self.n_jobs += 1
            if bad is not None:
                self.n_bad += 1
                result = TransferResult(job_id=result.job_id, success=False,
                                        transfer_size=result.transfer_size,
                                        transfer_time=result.transfer_time)
                logger.warning(
                    "OFFLOAD-NAN: store job %d has NaN/inf in its RAM copy (group %d, ref %d, dtype %s, page %d of %d, "
                    "byte offset %d); reported failed -> chunks dropped from the RAM tier (scan %.1f ms)",
                    result.job_id, *bad, dt * 1e3)
            self.done.put(result)
            now = time.monotonic()
            if now - self.last_log > 600:
                logger.info("KV-NANCHECK: %d store jobs, %.2f GiB scanned, %d failed, scan time %.1f s (since start)",
                            self.n_jobs, self.n_bytes / 2**30, self.n_bad, self.t_scan)
                self.last_log = now

    def _view(self, ptr: int, nbytes: int, utype) -> np.ndarray:
        return np.frombuffer((_ct.c_char * nbytes).from_address(ptr), dtype=utype)

    def _scan(self, job_id: int, ops):
        """ops: list of (dst ptrs ndarray, page bytes, group, ref). Returns None or the first bad page."""
        layout = _NC_LAYOUT
        if layout is None:
            return None
        for ptrs, page, g, r in ops:
            segs = layout[g][r]
            for i, base in enumerate(ptrs):
                for off, n, dt in segs:
                    m = _NC_MASKS.get(dt)
                    if m is None or n <= 0:
                        continue
                    utype, mask, pat = m
                    x = self._view(int(base) + off, n, utype)
                    if self.buf.nbytes < n:
                        self.buf = np.empty(n, dtype=np.uint8)
                    t = self.buf[:n].view(utype)
                    np.bitwise_and(x, mask, out=t)
                    self.n_bytes += n
                    if t.max() >= pat:
                        return (g, r, str(dt), i, len(ptrs), off)
        return None

    def shutdown(self):
        if self.th is not None:
            self.q.put(None)


# KV-RING load staging: with decoders running, each ring sub-batch of a load waits ~20 ms for its copy (the copy queues
# behind the decode steps' work), so a 1.8 GB prefix restore in 262 ring sub-batches took 5.5 s instead of 0.29 s.
# EXL3_KV_RING_LOAD_MB=N: loads stage through an N MB pinned buffer (two halves) instead of the ring: N/2-sized
# sub-batches, far fewer of them. Allocated on the first load, freed after EXL3_KV_RING_LOAD_IDLE_S (30) seconds
# without one (nothing stays pinned while idle). 0 = loads use the ring.
_LOAD_BYTES = int(float(_os.environ.get("EXL3_KV_RING_LOAD_MB", "0") or 0) * 2**20) // 4096 * 4096
_LOAD_IDLE_S = float(_os.environ.get("EXL3_KV_RING_LOAD_IDLE_S", "30") or 30)


class _Pacer:
    """KV-RING: EXL3_KV_RING_MB=N gives the pacer a pinned staging ring of N MB (two halves). Loads: the CPU copies a
    sub-batch's pages from the CPU tier into one half while the device DMAs the other half to the GPU; stores: the device
    DMAs into one half while the CPU copies the previous half out to the CPU tier. The CPU tier's memory is then never
    read or written by the device, so it needs no pinning or device mapping. 0 = off (direct copies).

    Paced RAM-tier copies: queues a transfer's copies in sub-batches of <= handler._pace_bytes on the transfer's own
    stream, waiting for each sub-batch to complete before queueing the next (see SingleDirectionOffloadingHandler
    .__init__). Runs in its own thread; all CUDA calls are stream-ordered after the waits transfer_async already put on
    the stream, so correctness is unchanged, only the copy engine's queue depth."""

    def __init__(self, handler):
        self.h = handler
        self.q: _queue.Queue = _queue.Queue()
        self.ev = None
        self.n_jobs = self.n_sub = 0
        self.ring_bytes = int(float(_os.environ.get("EXL3_KV_RING_MB", "0") or 0) * 2**20) // 4096 * 4096
        self.ring = None
        self.ring_stats = [0, 0.0, 0]   # bytes, seconds, jobs since the last log line
        self.log_every = int(_os.environ.get("EXL3_KV_RING_LOG_EVERY", "20") or 20)
        if self.ring_bytes > 0:
            self.ring = torch.empty(self.ring_bytes, dtype=torch.int8, pin_memory=PIN_MEMORY)
            self.ring_ev = [torch.Event(), torch.Event()]
            logger.info("KV-RING on: %.0f MB staging ring (pinned %s) for %s copies", self.ring_bytes / 2**20,
                        self.ring.is_pinned(), "GPU->CPU" if handler.gpu_to_cpu else "CPU->GPU")
        self.th = _threading.Thread(target=self._run, name="kv-pace", daemon=True)
        self.th.start()
        logger.info("KV-PACE on: tier copies queued in sub-batches of %.0f MB, each waited before the next (%s)",
                    handler._pace_bytes / 2**20, "GPU->CPU" if handler.gpu_to_cpu else "CPU->GPU")

    def submit(self, job_id, stream, fn, src, dst, sizes, order_any, end_event):
        self.q.put((job_id, stream, fn, src, dst, sizes, order_any, end_event))

    def shutdown(self):
        self.q.put(None)
        self.th.join(timeout=30)

    def _run(self):
        if self.ev is None:
            self.ev = torch.Event()
        while True:
            try:
                item = self.q.get(timeout=5.0)
            except _queue.Empty:
                big = getattr(self, "big", None)
                if big is not None and time.monotonic() - self.big_t > _LOAD_IDLE_S:
                    self.big = None                      # idle: give the pinned load staging buffer back
                    logger.info("KV-RING: load staging buffer (%.0f MB) freed after %.0f s idle", _LOAD_BYTES / 2**20,
                                _LOAD_IDLE_S)
                continue
            if item is None:
                return
            job_id, stream, fn, src, dst, sizes, order_any, end_event = item
            if self.ring is not None:
                try:
                    t0 = time.monotonic()
                    nb = self._run_ring(stream, fn, src, dst, sizes, order_any)
                    with current_platform.stream(stream):
                        end_event.record(stream)
                    self.n_jobs += 1
                    st = self.ring_stats
                    st[0] += nb; st[1] += time.monotonic() - t0; st[2] += 1
                    if st[2] >= self.log_every:
                        logger.info("KV-RING %s: %d jobs, %.2f GB in %.2f s (%.2f GB/s in copy time)",
                                    "store" if self.h.gpu_to_cpu else "load", st[2], st[0] / 1e9, st[1],
                                    st[0] / 1e9 / max(st[1], 1e-9))
                        self.ring_stats = [0, 0.0, 0]
                except Exception as e:
                    logger.error("KV-RING: job %d failed (%r); the transfer is reported failed", job_id, e)
                    self.h._failed.add(job_id)
                    try:
                        with current_platform.stream(stream):
                            end_event.record(stream)
                    except Exception:
                        pass
                finally:
                    self.h._pending_submit.discard(job_id)
                continue
            try:
                sz = sizes.numpy()
                n = len(sz)
                a = 0
                with current_platform.stream(stream):
                    while a < n:
                        b, acc = a, 0
                        while b < n and (b == a or acc + int(sz[b]) <= self.h._pace_bytes):
                            acc += int(sz[b]); b += 1
                        fn(src[a:b], dst[a:b], sizes[a:b], is_src_access_order_any=order_any)
                        self.ev.record(stream)
                        self.ev.synchronize()            # one sub-batch in the copy engine's queue at a time
                        self.n_sub += 1
                        a = b
                    end_event.record(stream)
                self.n_jobs += 1
            except Exception as e:  # never leave a job unrecorded: record the end so the engine does not hang
                logger.error("KV-PACE: job %d failed (%r); the transfer is reported finished", job_id, e)
                try:
                    with current_platform.stream(stream):
                        end_event.record(stream)
                except Exception:
                    pass
            finally:
                self.h._pending_submit.discard(job_id)

    def _run_ring(self, stream, fn, src, dst, sizes, order_any) -> int:
        """One transfer through the staging ring. Returns the bytes moved. Sub-batches fill one ring half (<= ring/2
        and <= the pace size); an op larger than a half is copied directly."""
        sz = sizes.numpy().astype(np.int64)
        S, D = src.numpy(), dst.numpy()
        n = len(sz)
        store = self.h.gpu_to_cpu
        half = self.ring_bytes // 2
        lim = min(half, self.h._pace_bytes) if self.h._pace_bytes > 0 else half
        base = self.ring.data_ptr()
        if _LOAD_BYTES > self.ring_bytes and not store:
            # a load (a request waits for it): the on-demand load staging buffer, fewer and larger sub-batches
            if getattr(self, "big", None) is None:
                _t = time.perf_counter()
                self.big = torch.empty(_LOAD_BYTES, dtype=torch.int8, pin_memory=PIN_MEMORY)
                logger.info("KV-RING: load staging buffer %.0f MB allocated (pinned %s, %.0f ms)", _LOAD_BYTES / 2**20,
                            self.big.is_pinned(), (time.perf_counter() - _t) * 1000)
            self.big_t = time.monotonic()
            base = self.big.data_ptr()
            half = lim = _LOAD_BYTES // 2
        batches, a = [], 0
        while a < n:
            b, acc = a, 0
            while b < n and acc + int(sz[b]) <= lim:
                acc += int(sz[b]); b += 1
            if b == a:
                batches.append((a, a + 1, True)); a += 1
            else:
                batches.append((a, b, False)); a = b
        desc = [torch.empty(n, dtype=src.dtype, pin_memory=PIN_MEMORY) for _ in range(2)]
        ev = self.ring_ev
        used = [None, None]     # per half: (a, b) of the batch whose DMA uses it (stores: still to copy out)
        memmove = _ct.memmove
        with current_platform.stream(stream):
            for k, (a, b, direct) in enumerate(batches):
                h = k % 2
                if used[h] is not None:
                    ev[h].synchronize()
                    if store:
                        pa, pb = used[h]
                        off = base + h * half
                        for i in range(pa, pb):
                            memmove(int(D[i]), off, int(sz[i])); off += int(sz[i])
                    used[h] = None
                if direct:
                    fn(src[a:b], dst[a:b], sizes[a:b], is_src_access_order_any=order_any)
                    ev[h].record(stream)
                    ev[h].synchronize()
                    continue
                off = base + h * half
                d = desc[h][: b - a]
                dn = d.numpy()
                if store:
                    for i in range(a, b):
                        dn[i - a] = off; off += int(sz[i])
                    fn(src[a:b], d, sizes[a:b], is_src_access_order_any=order_any)
                else:
                    for i in range(a, b):
                        memmove(off, int(S[i]), int(sz[i])); dn[i - a] = off; off += int(sz[i])
                    fn(d, dst[a:b], sizes[a:b], is_src_access_order_any=order_any)
                ev[h].record(stream)
                used[h] = (a, b)
                self.n_sub += 1
            # drain: the other half first (it is older), then this one
            last = (len(batches) - 1) % 2 if batches else 0
            for h in (1 - last, last):
                if used[h] is not None:
                    ev[h].synchronize()
                    if store:
                        pa, pb = used[h]
                        off = base + h * half
                        for i in range(pa, pb):
                            memmove(int(D[i]), off, int(sz[i])); off += int(sz[i])
                    used[h] = None
        return int(sz.sum())


def _select_swap_blocks_fn(
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
    gpu_to_cpu: bool,
):
    """Resolve the swap_blocks function for a handler at init time."""
    # GPU->CPU is bandwidth-bound; the dedicated copy engine beats Triton.
    if gpu_to_cpu:
        return ops.swap_blocks_batch
    # Fall back to the C++ DMA path on platforms where Triton isn't usable
    # (e.g. ROCm builds without Triton) or where GPU kernels cannot directly
    # dereference CPU pointers (XPU lacks CUDA's unified virtual address space,
    # so the Triton kernel's tl.load(cpu_ptr) is invalid on XPU).
    if not HAS_TRITON or current_platform.is_xpu():
        return ops.swap_blocks_batch
    page_sizes = [r.page_size_bytes for g in kv_cache_groups_data_refs for r in g]
    # Triton wins only on small, 8-byte-aligned payloads.
    if (
        not page_sizes
        or max(page_sizes) >= THRESHOLD_BYTES
        or any(s % 8 for s in page_sizes)
    ):
        return ops.swap_blocks_batch
    chunk = min(triton.next_power_of_2(max(page_sizes)), 8192)
    return functools.partial(swap_blocks_batch, bytes_per_chunk=chunk)


@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    batch_src: torch.Tensor
    batch_dst: torch.Tensor
    batch_sizes: torch.Tensor


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    blocks_per_chunk: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
):
    """
    Compute byte pointers for sub-blocks of the given block IDs.

    Each block in block_ids contains blocks_per_chunk sub-blocks.
    The pointer for sub-block j of block b is:
        base_ptr + b * row_stride + j * block_page_size

    where block_page_size = tensor.shape[1] // blocks_per_chunk (gpu page size).

    This handles tensors where row_stride != blocks_per_chunk * block_page_size
    (e.g. non-contiguous CPU tensors).

    Args:
        block_ids: array of block IDs at the tensor's native granularity.
        blocks_per_chunk: number of sub-blocks per block.
        output: pre-allocated pointer array to write pointers into.
        tensor: the source or destination tensor.
        skip_count: sub-blocks to skip in the first block.
    """
    assert skip_count < blocks_per_chunk

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if blocks_per_chunk == 1:
        # Fast path: 1:1 mapping, no sub-block expansion needed.
        output[:] = base_ptr + block_ids.astype(np.uint64)[:num_sub_blocks] * row_stride
        return

    # Vectorized expansion for blocks_per_chunk > 1.
    assert tensor.shape[1] % blocks_per_chunk == 0
    block_page_size = tensor.shape[1] // blocks_per_chunk
    sub_offsets = np.arange(blocks_per_chunk, dtype=np.uint64) * block_page_size
    # (num_blocks, 1) + (1, blocks_per_chunk) -> (num_blocks, blocks_per_chunk)
    all_ptrs = (
        base_ptr + block_ids.astype(np.uint64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    # Flatten and apply skip_count / truncation
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


def pin_mmap_region(region: SharedOffloadRegion) -> None:
    """Register the entire mmap as CUDA pinned memory via cudaHostRegister."""
    if not current_platform.is_cuda_alike():
        logger.info(
            "Skipping mmap host registration on %s; cudaHostRegister is only "
            "available on CUDA/ROCm.",
            current_platform.device_name,
        )
        return

    rank = region.rank

    base_ptr = region._base.data_ptr()
    result = torch.cuda.cudart().cudaHostRegister(base_ptr, region.total_size_bytes, 0)
    if result.value != 0:
        logger.warning(
            "cudaHostRegister failed for rank=%d (code=%d) — "
            "transfers will still work but may be slower (unpinned DMA)",
            rank,
            result,
        )
    else:
        logger.debug(
            "cudaHostRegister rank=%d %.2f GB",
            rank,
            region.total_size_bytes / 1e9,
        )
        region.is_pinned = True


def _new_descriptor_buffers(
    num_copy_ops: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pin = PIN_MEMORY
    # CUDA cache_kernels.cu requires int64; XPU DMA engine requires uint64.
    ptr_dtype = torch.uint64 if current_platform.is_xpu() else torch.int64
    return (
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
    )


class SingleDirectionOffloadingHandler:
    """
    Handles transfers for a single direction, either CPU->GPU or GPU->CPU.
    Transfers are guaranteed to be executed in order of their submission.
    Each transfer uses a unique CUDA stream, and its stream will start
    executing only after the streams of previous transfers have finished.
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        blocks_per_chunk: int,
        kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        mmap_region: SharedOffloadRegion | None = None,
    ):
        """
        Initialize a SingleDirectionOffloadingHandler.

        Args:
            gpu_tensors: list of GPU KV cache tensors.
                Each of shape (num_gpu_blocks, gpu_page_size_bytes) with dtype int8.
            cpu_tensors: list of CPU KV cache tensors.
                Each of shape (num_cpu_blocks, cpu_page_size_bytes) with dtype int8.
                Order should match gpu_tensors.
            kv_cache_groups_data_refs: list of CanonicalKVCacheRef per group.
            gpu_to_cpu: if True, transfer from GPU to CPU; otherwise CPU to GPU.
        """
        assert len(gpu_tensors) == len(cpu_tensors)
        assert len(gpu_tensors) > 0

        # assert input tensors are as expected
        for gpu_tensor, cpu_tensor in zip(gpu_tensors, cpu_tensors):
            assert gpu_tensor.dtype == torch.int8
            assert gpu_tensor.ndim == 2
            assert gpu_tensor.is_cuda or gpu_tensor.is_xpu
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            _, gpu_page_size = gpu_tensor.shape
            _, cpu_page_size = cpu_tensor.shape
            assert cpu_page_size == gpu_page_size * blocks_per_chunk

        self.src_tensors: list[torch.Tensor] = (
            gpu_tensors if gpu_to_cpu else cpu_tensors
        )
        self.dst_tensors: list[torch.Tensor] = (
            cpu_tensors if gpu_to_cpu else gpu_tensors
        )
        self.gpu_to_cpu: bool = gpu_to_cpu
        self.kv_cache_groups_data_refs = kv_cache_groups_data_refs
        self._swap_blocks_batch = _select_swap_blocks_fn(
            kv_cache_groups_data_refs, gpu_to_cpu
        )

        # GPU blocks may be smaller
        # cpu_page_size = gpu_page_size * blocks_per_chunk.
        self.src_blocks_per_chunk = 1 if self.gpu_to_cpu else blocks_per_chunk
        self.dst_blocks_per_chunk = blocks_per_chunk if self.gpu_to_cpu else 1

        # mmap_region to clean up on shutdown (gpu_to_cpu handler owns it)
        self._mmap_region = mmap_region
        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # queue of transfers (job_id, stream, event)
        self._transfers: deque[Transfer] = deque()
        # list of CUDA streams available for re-use
        self._stream_pool: list[torch.cuda.Stream] = []
        # list of CUDA events available for re-use
        self._event_pool: list[torch.Event] = []
        # list of pinned descriptor buffer sets available for re-use
        self._buffer_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._nanchk = _NanChecker() if gpu_to_cpu else None
        self._nc_ops: dict[int, list] = {}
        # Paced RAM-tier copies: EXL3_KV_COPY_PACE_MB=N
        # submits a transfer's copies in sub-batches of <= N MB from a helper thread that waits for each sub-batch to
        # finish before queueing the next, so the copy engine's queue never holds more than one sub-batch of tier
        # traffic ahead of the decode steps' own small copies (a 58.8 MB block takes 34 ms on the 5090's Gen2 x4 link; the
        # whole batch of a hit, several hundred MB, queued at once stalled decode steps by ~16 ms on average at 8 agents).
        # 0 = off (upstream: one cuMemcpyBatchAsync per transfer).
        self._pace_bytes = int(float(_os.environ.get("EXL3_KV_COPY_PACE_MB", "0") or 0) * 2**20)
        self._pending_submit: set[int] = set()        # jobs whose copies the pacer has not fully queued yet
        self._failed: set[int] = set()                # KV-RING: jobs whose host copies failed (reported success=False)
        self._pacer = _Pacer(self) if self._pace_bytes > 0 else None

    def transfer_async(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        num_src_blocks = len(src_blocks)
        num_dst_blocks = len(dst_blocks)

        # There are 2 types of transfers:
        # 1. GPU -> CPU
        # 2. CPU -> GPU
        #
        # transfers are also to CPU blocks, EXCEPT MAYBE for the first and last block.
        # i.e. the first and last CPU blocks in src_blocks can match against
        # a smaller (byte-wise) set of GPU blocks in dst_blocks.
        # In such cases, we may need to skip some gpu-sized sub-blocks,
        # and start reading/writing from the middle of the first CPU block.
        # If we have multiple KV cache groups (when using HMA with hybrid models),
        # we may have a partial first/last CPU block per each group.
        # The group_sizes parameter encodes the size of each group of blocks
        # in the GPU dst_blocks.
        # If group_sizes is None, we assume all blocks belong to a single group.
        # The logical_offset parameter maps each group of blocks to its logical
        # offset inside the request, counting in GPU blocks.
        # This allows us to find the correct starting position
        # in the matching first CPU block.

        # extract group_sizes from the GPU spec
        gpu_spec = src_spec if self.gpu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self.kv_cache_groups_data_refs)

        # extract block indices from the GPU spec
        block_indices = gpu_spec.block_indices
        assert len(block_indices) == len(self.kv_cache_groups_data_refs)

        num_copy_ops = 0
        for group_size, group_data_refs in zip(
            group_sizes, self.kv_cache_groups_data_refs
        ):
            num_copy_ops += group_size * len(group_data_refs)

        # reuse a pooled buffer set, growing it if this transfer needs more room
        batch_src, batch_dst, batch_sizes = (
            self._buffer_pool.pop()
            if self._buffer_pool
            else _new_descriptor_buffers(num_copy_ops)
        )
        if batch_src.numel() < num_copy_ops:
            batch_src, batch_dst, batch_sizes = _new_descriptor_buffers(num_copy_ops)

        src = batch_src[:num_copy_ops]
        dst = batch_dst[:num_copy_ops]
        sizes = batch_sizes[:num_copy_ops]
        all_src = src.numpy()
        all_dst = dst.numpy()
        all_sizes = sizes.numpy()

        src_offset = 0
        dst_offset = 0
        op_idx = 0
        # count total number of bytes copied
        num_transfer_bytes = 0
        _nc_ranges = []  # (op start, op end, group, ref)
        for _nc_g, (group_size, block_idx, group_data_refs) in enumerate(zip(
            group_sizes, block_indices, self.kv_cache_groups_data_refs
        )):
            if group_size == 0:
                continue

            src_logical_blocks_to_skip = block_idx % self.src_blocks_per_chunk
            dst_logical_blocks_to_skip = block_idx % self.dst_blocks_per_chunk
            src_logical_blocks_count = group_size + src_logical_blocks_to_skip
            dst_logical_blocks_count = group_size + dst_logical_blocks_to_skip

            dst_blocks_count = cdiv(dst_logical_blocks_count, self.dst_blocks_per_chunk)
            dst_end_offset = dst_offset + dst_blocks_count
            assert dst_end_offset <= num_dst_blocks

            src_blocks_count = cdiv(src_logical_blocks_count, self.src_blocks_per_chunk)
            src_end_offset = src_offset + src_blocks_count
            assert src_end_offset <= num_src_blocks

            group_src = src_blocks[src_offset:src_end_offset]
            group_dst = dst_blocks[dst_offset:dst_end_offset]

            for _nc_r, data_ref in enumerate(group_data_refs):
                t_idx = data_ref.tensor_idx
                end_idx = op_idx + group_size
                _nc_ranges.append((op_idx, end_idx, _nc_g, _nc_r, data_ref.page_size_bytes))

                compute_sub_block_ptrs(
                    group_src,
                    self.src_blocks_per_chunk,
                    all_src[op_idx:end_idx],
                    self.src_tensors[t_idx],
                    skip_count=src_logical_blocks_to_skip,
                )
                compute_sub_block_ptrs(
                    group_dst,
                    self.dst_blocks_per_chunk,
                    all_dst[op_idx:end_idx],
                    self.dst_tensors[t_idx],
                    skip_count=dst_logical_blocks_to_skip,
                )

                all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
                num_transfer_bytes += group_size * data_ref.page_size_bytes
                op_idx = end_idx

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        assert src_offset == num_src_blocks
        assert dst_offset == num_dst_blocks
        assert op_idx == num_copy_ops

        stream = (
            self._stream_pool.pop() if self._stream_pool else current_platform.Stream()
        )
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

        if self.gpu_to_cpu:
            # wait for model computation to finish before offloading
            stream.wait_stream(current_platform.current_stream())
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            # assure job will start only after the previous one completes
            stream.wait_event(last_event)
        # CPU->GPU reads from host pinned memory, which is never written
        # by a concurrent GPU stream, so CU_MEMCPY_SRC_ACCESS_ORDER_ANY is
        # safe and lets the driver pipeline source reads. GPU->CPU reads
        # from the live GPU KV cache, which the compute stream keeps
        # writing; we must keep STREAM ordering so source reads are gated
        # by the transfer stream's wait_stream(compute) barrier.
        is_src_access_order_any = not self.gpu_to_cpu
        if self._pacer is not None and num_copy_ops > 0:
            # the GDN-neutral rewrite (vllm_tier._NeutralCopy, wrapped around _swap_blocks_batch)
            # must run here, inside its transfer_async window and on the whole op list; the pacer then calls the real
            # swap on sub-slices from its thread. end_event is recorded by the pacer after the last sub-batch.
            fn = self._swap_blocks_batch
            nc = getattr(fn, "__self__", None)
            if nc is not None and hasattr(nc, "rewrite") and hasattr(nc, "orig_swap"):
                new = nc.rewrite(src.numpy(), dst.numpy(), sizes.numpy())
                if new is not None:
                    pin = src.is_pinned()
                    src, dst, sizes = (torch.from_numpy(x).pin_memory() if pin else torch.from_numpy(x) for x in new)
                    nc.inflight[job_id] = (src, dst, sizes)
                fn = nc.orig_swap
            with current_platform.stream(stream):
                start_event.record(stream)
            self._pending_submit.add(job_id)
            self._pacer.submit(job_id, stream, fn, src, dst, sizes, is_src_access_order_any, end_event)
        else:
            with current_platform.stream(stream):
                start_event.record(stream)
                if num_copy_ops > 0:
                    self._swap_blocks_batch(
                        src,
                        dst,
                        sizes,
                        is_src_access_order_any=is_src_access_order_any,
                    )
                end_event.record(stream)

        self._transfer_events[job_id] = end_event
        if self._nanchk is not None and self._nanchk.on:
            self._nc_ops[job_id] = _nc_ranges
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
                batch_src=batch_src,
                batch_dst=batch_dst,
                batch_sizes=batch_sizes,
            )
        )

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        # an event not yet recorded reports query() == True, so a job still being queued by the
        # pacer must not count as finished
        while (self._transfers and self._transfers[0].job_id not in self._pending_submit
               and self._transfers[0].end_event.query()):
            transfer = self._transfers.popleft()
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )  # elapsed_time is in milliseconds
            result = TransferResult(
                job_id=transfer.job_id,
                success=transfer.job_id not in self._failed,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
            )
            self._failed.discard(transfer.job_id)

            ranges = self._nc_ops.pop(transfer.job_id, None)
            if ranges is not None:
                # copy the destination page pointers before the descriptor
                # buffers go back to the pool; the result is reported once the scan has finished.
                dst_all = transfer.batch_dst.numpy()
                nc_ops = [(dst_all[a:b].copy(), page, g, r) for a, b, g, r, page in ranges]
                self._nanchk.q.put((result, nc_ops))
            else:
                results.append(result)
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            self._buffer_pool.append(
                (transfer.batch_src, transfer.batch_dst, transfer.batch_sizes)
            )
            del self._transfer_events[transfer.job_id]
        if self._nanchk is not None:
            while True:
                try:
                    results.append(self._nanchk.done.get_nowait())
                except _queue.Empty:
                    break
        return results

    def wait(self, job_ids: set[int]):
        for job_id in job_ids:
            while job_id in self._pending_submit:       # let the pacer queue the rest first
                time.sleep(0.0005)
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def shutdown(self) -> None:
        if self._nanchk is not None:
            self._nanchk.shutdown()
        if self._pacer is not None:
            self._pacer.shutdown()
        while self._transfers:
            transfer = self._transfers.popleft()
            transfer.end_event.synchronize()
        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self._buffer_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()
        if self._mmap_region is not None:
            self._mmap_region.cleanup()
            self._mmap_region = None


class CPUOffloadingWorker(OffloadingWorker):
    """OffloadingWorker for CPU offloading.

    Composes two SingleDirectionOffloadingHandler instances (one for each
    direction) and exposes them through the explicit submit_store /
    submit_load API.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_blocks: int,
        mmap_region: SharedOffloadRegion | None = None,
    ):
        pin_memory = PIN_MEMORY
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region)

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * blocks_per_chunk

            if mmap_region is not None:
                cpu_tensor = mmap_region.create_next_view(cpu_page_size_bytes)
            else:
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_blocks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                logger.debug(
                    "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_blocks,
                    cpu_page_size_bytes,
                    num_cpu_blocks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)

        self._store_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            mmap_region=mmap_region,
        )

        self._load_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=False,
        )

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """Async GPU -> CPU."""
        return self._store_handler.transfer_async(job_id, src_spec, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        """Async CPU -> GPU."""
        return self._load_handler.transfer_async(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        return self._store_handler.get_finished() + self._load_handler.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    def shutdown(self) -> None:
        self._store_handler.shutdown()
        self._load_handler.shutdown()
