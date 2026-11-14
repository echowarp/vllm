# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests: hybrid (attention + Mamba/GDN) models in the cross-layer (packed) KV layout.

Small fake hybrid configs (int4_per_token_head attention pages with inline scales, TRITON_ATTN NHD; GDN groups with
a bf16 conv state + fp32 SSM state padded to the attention page) are planned with vLLM's own planner, unpacked and
packed ("enable_cross_layers_blocks": "true"), then run through:

  views   : GPUModelRunner._reshape_kv_cache_tensors: every layer's view of every block is written and read back;
            packed and unpacked hold identical bytes per (block, slot) and identical contents per (layer, block).
  zeroer  : KVBlockZeroer (Triton interpreter): zeroing new blocks changes only those blocks' pages, in both layouts.
  offload : OffloadingConnectorWorker.register_kv_caches: the same canonical tensors, page sizes, group refs and
            contents in both layouts; per-block address = base + slot x page + block x block_stride.
  prod    : a Qwen3-Next-sized geometry on meta tensors (strides, offsets, bounds, zeroer parameters, offload refs).

Run on CPU: TRITON_INTERPRET=1 VLLM_KV_CACHE_LAYOUT=NHD pytest tests/v1/worker/test_packed_kv_hybrid.py
"""
import ast
import os
import types

os.environ.setdefault("VLLM_KV_CACHE_LAYOUT", "NHD")
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

import vllm.triton_utils as _tu  # noqa: E402

# real Triton for the zeroer kernel (vLLM stubs it without a GPU driver); TRITON_INTERPRET=1 runs it on CPU
_tu.triton, _tu.tl = triton, tl

from vllm.utils.torch_utils import get_dtype_size  # noqa: E402
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend  # noqa: E402
from vllm.v1.core import kv_cache_utils as kcu  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    AttentionSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVQuantMode,
    MambaSpec,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache  # noqa: E402


def _runner_path():
    import importlib.util

    return importlib.util.find_spec("vllm.v1.worker.gpu_model_runner").origin


def extract_funcs(path, names, glb, cls=None):
    """Exec the named function defs of a file (class-level if cls) in glb, without importing the module (the GPU
    model runner module pulls in device code)."""
    tree = ast.parse(open(path).read())
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    out = {}
    for n in body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            n.decorator_list = []
            exec(compile(ast.Module(body=[n], type_ignores=[]), path, "exec"), glb)
            out[n.name] = glb[n.name]
    missing = set(names) - set(out)
    assert not missing, f"{path}: functions not found: {missing}"
    return out


# ------------------------------------------------------------------------------------------------ fake model config
BS = 16          # tokens per block (manager block == kernel block)
H = 1            # KV heads per attention layer
HEAD = 256
CONV = (6, 32)   # conv state rows (3 history + 3 spec window) x conv dim, bf16
SSM = (2, 8, 8)  # SSM state heads x k x v, fp32


def attn_spec():
    return FullAttentionSpec(block_size=BS, num_kv_heads=H, head_size=HEAD, dtype=torch.uint8,
                             kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD)


def gdn_spec(page):
    return MambaSpec(block_size=BS, shapes=(CONV, SSM), dtypes=(torch.bfloat16, torch.float32),
                     page_size_padded=page)


CONFIGS = {
    # 3 attention layers + 1 GDN group of 2 (slot 2 holds only an attention layer: a padding slot)
    "A_3attn_1gdn": (3, [2]),
    # 3 attention layers + 2 GDN groups of 3 (Qwen3-Next-style structure in small)
    "C_3attn_2gdn": (3, [3, 3]),
    # GDN groups first, attention last (the order the planner produces for these hybrid models)
    "D_prodorder_2gdn_3attn": (3, [2, 2], True),
}
NUM_BLOCKS = 8


def fake_vllm_config(packed):
    extra = {}
    if packed:
        extra["enable_cross_layers_blocks"] = "true"
    return types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(kv_connector_extra_config=extra),
        cache_config=types.SimpleNamespace(num_gpu_blocks_override=None))


def make_groups(n_attn, gdn_sizes, attn_last=False, a=None, gdn=None):
    a = a or attn_spec()
    page = a.page_size_bytes
    attn = KVCacheGroupSpec([f"model.layers.{i}.self_attn.attn" for i in range(n_attn)], a)
    groups = []
    li = 100
    for g in gdn_sizes:
        names = [f"model.layers.{li + j}.linear_attn" for j in range(g)]
        li += 100
        groups.append(KVCacheGroupSpec(names, (gdn or gdn_spec)(page)))
    return groups + [attn] if attn_last else [attn] + groups


def plan(groups, packed, nblocks=None):
    page = groups[0].kv_cache_spec.page_size_bytes
    nslots = max(len(g.layer_names) for g in groups)
    avail = (nblocks or NUM_BLOCKS) * page * nslots + page // 2   # not a multiple: planners must floor the same way
    return kcu.get_kv_cache_config_from_groups(fake_vllm_config(packed), groups, avail)


class Group:
    """AttentionGroup stand-in (the fields the runner, zeroer and offload code read)."""

    def __init__(self, gid, g):
        self.kv_cache_group_id = gid
        self.layer_names = g.layer_names
        self.kv_cache_spec = g.kv_cache_spec
        self.backend = TritonAttentionBackend if isinstance(g.kv_cache_spec, AttentionSpec) else None


RUNNER_FUNCS = ["_reshape_kv_cache_tensors", "_has_mixed_attention_kv_layout", "_update_hybrid_attention_mamba_layout"]


def runner_class(path=None):
    path = path or _runner_path()
    glb = dict(torch=torch, AttentionSpec=AttentionSpec, MambaSpec=MambaSpec, KVQuantMode=KVQuantMode,
               get_dtype_size=get_dtype_size, _reshape_attention_kv_cache=_reshape_attention_kv_cache)
    fns = extract_funcs(path, RUNNER_FUNCS, glb, cls="GPUModelRunner")

    class FakeRunner:
        def __init__(self, cfg):
            self.kv_cache_config = cfg
            self.runner_only_attn_layers = set()
            self.cache_config = types.SimpleNamespace(cache_dtype="int4_per_token_head")
            self.groups = [Group(i, g) for i, g in enumerate(cfg.kv_cache_groups)]

        def _kv_cache_spec_attn_group_iterator(self):
            yield from self.groups

    for k, f in fns.items():
        setattr(FakeRunner, k, f)
    return FakeRunner


def allocate(cfg, device="cpu"):
    """= gpu_model_runner._allocate_kv_cache_tensors (CPU, or meta for production geometry)"""
    raw, packed_backing, backings = {}, None, []
    for t in cfg.kv_cache_tensors:
        if t.block_stride > 0:
            if packed_backing is None:
                packed_backing = torch.zeros(t.size, dtype=torch.int8, device=device)
                backings.append(packed_backing)
            tensor = packed_backing
        else:
            tensor = torch.zeros(t.size, dtype=torch.int8, device=device)
            backings.append(tensor)
        for ln in t.shared_by:
            raw[ln] = tensor
    return raw, backings


def build(groups, packed, device="cpu", nblocks=None):
    cfg = plan(groups, packed, nblocks)
    R = runner_class()
    r = R(cfg)
    raw, backings = allocate(cfg, device)
    kv = r._reshape_kv_cache_tensors(raw, [BS] * len(cfg.kv_cache_groups))
    return types.SimpleNamespace(cfg=cfg, runner=r, kv=kv, backings=backings)


def slot_pages(L, cfg):
    """[(slot k, int8 (N, P) view of that slot's pages)] for either layout"""
    page = cfg.kv_cache_groups[0].kv_cache_spec.page_size_bytes
    N = cfg.num_blocks
    out = []
    for k, t in enumerate(cfg.kv_cache_tensors):
        if t.block_stride:
            b = L.backings[0]
            out.append(torch.as_strided(b, (N, page), (t.block_stride, 1), t.offset))
        else:
            out.append(L.backings[k].view(N, page))
    return out


def fill_value(layer_idx, block, shape, dtype):
    g = torch.Generator().manual_seed(layer_idx * 1000 + block)
    if dtype in (torch.uint8, torch.int8):
        return torch.randint(0, 255, shape, generator=g, dtype=torch.int32).to(dtype)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype)


def owner(b, groups):
    """block ids are global (one BlockPool): block b belongs to one group at a time; here b % groups"""
    return b % len(groups)


def write_all(L, groups):
    """write every layer of every block's owning group through the views (block 0 = null block skipped, as in
    vLLM); returns expectations"""
    exp = {}
    names = [(gi, n) for gi, g in enumerate(groups) for n in g.layer_names]
    for li, (gi, n) in enumerate(names):
        v = L.kv[n]
        for b in range(1, L.cfg.num_blocks):
            if owner(b, groups) != gi:
                continue
            if isinstance(v, torch.Tensor):
                x = fill_value(li, b, tuple(v[b].shape), v.dtype)
                v[b].copy_(x)
                exp[(n, b, 0)] = x
            else:
                for si, st in enumerate(v):
                    x = fill_value(li * 10 + si, b, tuple(st[b].shape), st.dtype)
                    st[b].copy_(x)
                    exp[(n, b, si)] = x
    return exp


def read_ok(L, exp):
    bad = []
    for (n, b, si), x in exp.items():
        v = L.kv[n]
        got = v[b] if isinstance(v, torch.Tensor) else v[si][b]
        if not torch.equal(got.contiguous().view(torch.uint8), x.contiguous().view(torch.uint8)):
            bad.append((n, b, si))
    return bad


# ------------------------------------------------------------------------------------------------ tests
CASES = [pytest.param(make_groups(na, gs, *last), id=c) for c, (na, gs, *last) in CONFIGS.items()]


@pytest.mark.parametrize("groups", CASES)
def test_plan(groups):
    u, p = plan(groups, False), plan(groups, True)
    assert p.num_blocks == u.num_blocks == NUM_BLOCKS
    assert [t.shared_by for t in u.kv_cache_tensors] == [t.shared_by for t in p.kv_cache_tensors]
    page = groups[0].kv_cache_spec.page_size_bytes
    assert all(t.block_stride == page * len(p.kv_cache_tensors) and t.offset == k * page
               for k, t in enumerate(p.kv_cache_tensors))


@pytest.mark.parametrize("groups", CASES)
def test_views(groups):
    U = build(groups, False)
    P = build(groups, True)
    eu, ep = write_all(U, groups), write_all(P, groups)
    # unpacked: the previous formula (stride(0) = page, state views from page offset 0)
    page = groups[0].kv_cache_spec.page_size_bytes
    for n, v in U.kv.items():
        if isinstance(v, (list, tuple)):
            assert v[0].storage_offset() == 0
            assert all(t.stride(0) * t.element_size() == page for t in v)
    assert not read_ok(P, ep), "packed views alias"
    assert not read_ok(U, eu)
    for (n, b, si) in eu:
        pv = P.kv[n] if isinstance(P.kv[n], torch.Tensor) else P.kv[n][si]
        uv = U.kv[n] if isinstance(U.kv[n], torch.Tensor) else U.kv[n][si]
        assert torch.equal(pv[b].contiguous().view(torch.uint8), uv[b].contiguous().view(torch.uint8))
    assert all(torch.equal(a, b) for a, b in zip(slot_pages(P, P.cfg), slot_pages(U, U.cfg)))
    bs = P.cfg.kv_cache_tensors[0].block_stride
    for v in P.kv.values():
        for t in (v if isinstance(v, (list, tuple)) else [v]):
            assert t.stride(0) * t.element_size() == bs


def _zeroer(L):
    from vllm.v1.worker.utils import KVBlockZeroer

    fwd = {n: types.SimpleNamespace(kv_cache=v) for n, v in L.kv.items()}
    return KVBlockZeroer(torch.device("cpu"), pin_memory=False, attn_groups_iter=iter(L.runner.groups),
                         kernel_block_sizes=[BS] * len(L.cfg.kv_cache_groups), cache_dtype="int4_per_token_head",
                         static_forward_context=fwd)


@pytest.mark.parametrize("groups", CASES)
def test_zeroer(groups):
    blocks = [2, 5, NUM_BLOCKS - 1]
    res = {}
    for packed in (False, True):
        L = build(groups, packed)
        for b in L.backings:
            b.fill_(0x5A)
        before = [s.clone() for s in slot_pages(L, L.cfg)]
        z = _zeroer(L)
        z.zero_block_ids(blocks)
        after = slot_pages(L, L.cfg)
        assert all(bool((a[blocks] == 0).all()) for a in after)
        others = [i for i in range(NUM_BLOCKS) if i not in blocks]
        assert all(torch.equal(a[others], b4[others]) for a, b4 in zip(after, before))
        nz = sum(int((b == 0).sum()) for b in L.backings)
        page = L.cfg.kv_cache_groups[0].kv_cache_spec.page_size_bytes
        assert nz == len(blocks) * page * len(L.cfg.kv_cache_tensors)
        if packed:
            assert z._block_stride_el * 4 == L.cfg.kv_cache_tensors[0].block_stride
        else:
            assert z._block_stride_el == 0
        res[packed] = after
    assert all(torch.equal(a, b) for a, b in zip(res[True], res[False]))


class FakeSpec:
    def __init__(self):
        self.caches = None

    def get_worker(self, kv_caches):
        self.caches = kv_caches
        return object()


def register(L):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import OffloadingConnectorWorker

    try:
        import vllm.v1.kv_offload.cpu.gpu_worker as gw
        gw._NC_LAYOUT = None
    except ImportError:
        gw = None
    spec = FakeSpec()
    w = OffloadingConnectorWorker(spec, L.cfg)
    w.register_kv_caches(L.kv)
    return spec.caches, getattr(gw, "_NC_LAYOUT", None)


def _desc(c, strides=False):
    return ([(tuple(t.tensor.shape),) + ((t.tensor.stride(),) if strides else ()) + (t.page_size_bytes,)
             for t in c.tensors],
            [[(r.tensor_idx, r.page_size_bytes) for r in g] for g in c.group_data_refs])


@pytest.mark.parametrize("groups", CASES)
def test_offload(groups):
    U = build(groups, False)
    P = build(groups, True)
    write_all(U, groups)
    write_all(P, groups)
    cu, nu = register(U)
    cp, np_ = register(P)
    page = groups[0].kv_cache_spec.page_size_bytes
    assert _desc(cp) == _desc(cu)
    assert len(cp.tensors) == len(cu.tensors)
    assert all(torch.equal(a.tensor, b.tensor) for a, b in zip(cp.tensors, cu.tensors))
    bs = P.cfg.kv_cache_tensors[0].block_stride
    base_ptr = P.backings[0].data_ptr()
    for k, t in enumerate(cp.tensors):
        x = t.tensor
        assert x.stride(0) == bs and x.data_ptr() == base_ptr + k * page
        y = x.view(torch.int8).view((-1, t.page_size_bytes))
        assert y.data_ptr() == x.data_ptr() and y.stride() == x.stride()
    assert np_ == nu   # page-relative NaN-check segments (None if the worker has no such layout)


def test_prod_geometry():
    """Qwen3-Next-sized geometry on meta tensors: 3 GDN groups x 16 layers (conv (10, 10240) fp16 + SSM
    (48, 128, 128) fp32, padded to the attention page) + 17 int4_per_token_head attention layers (4 KV heads x 3200
    tokens x 264 B), 146 blocks."""
    global BS
    a = FullAttentionSpec(block_size=3200, num_kv_heads=4, head_size=256, dtype=torch.uint8,
                          kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD)
    P = a.page_size_bytes

    def gdn(page):
        return MambaSpec(block_size=3200, shapes=((10, 10240), (48, 128, 128)), dtypes=(torch.float16, torch.float32),
                         page_size_padded=page)

    groups = make_groups(17, [16, 16, 16], True, a=a, gdn=gdn)
    old_bs, BS = BS, 3200
    try:
        N = 146
        U = build(groups, False, "meta", N)
        Pk = build(groups, True, "meta", N)
        bs = Pk.cfg.kv_cache_tensors[0].block_stride
        assert P == 3379200 and bs == 17 * P and Pk.cfg.num_blocks == U.cfg.num_blocks == N
        slot_of = {ln: k for k, t in enumerate(Pk.cfg.kv_cache_tensors) for ln in t.shared_by}
        total = N * bs
        for n, v in Pk.kv.items():
            k = slot_of[n]
            vu = U.kv[n]
            ts, tu = (v, vu) if isinstance(v, (list, tuple)) else ([v], [vu])
            for t, x in zip(ts, tu):
                el = t.element_size()
                end = (sum((s - 1) * st for s, st in zip(t.shape, t.stride())) + 1) * el + t.storage_offset() * el
                assert t.shape == x.shape and t.stride()[1:] == x.stride()[1:]
                assert t.stride(0) * el == bs and x.stride(0) * el == P
                assert t.storage_offset() * el - k * P == x.storage_offset() * el
                assert end <= total
        att = Pk.kv["model.layers.0.self_attn.attn"]
        assert att.stride(0) == bs and tuple(att.shape)[0] == N
        from vllm.v1.worker.utils import KVBlockZeroer
        fwd = {n: types.SimpleNamespace(kv_cache=v) for n, v in Pk.kv.items()}
        z = KVBlockZeroer(torch.device("cpu"), pin_memory=False, attn_groups_iter=iter(Pk.runner.groups),
                          kernel_block_sizes=[3200] * 4, cache_dtype="int4_per_token_head", static_forward_context=fwd)
        seg, page_el, blk, nseg = z._meta
        assert nseg == 17 and page_el * 4 == P and z._block_stride_el * 4 == bs
        cu, nu = register(U)
        cp, np_ = register(Pk)
        du, dp = _desc(cu, True), _desc(cp, True)
        assert len(cp.tensors) == 17 and du[1] == dp[1]
        assert [x[0] for x in du[0]] == [x[0] for x in dp[0]]
        assert all(x[1] == (bs, 1) for x in dp[0])
        assert np_ == nu
    finally:
        BS = old_bs
