# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for vllm.v1.extension_hooks and its call sites."""

import contextlib
import json

import pytest

from vllm.v1 import extension_hooks as hooks
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.request import RequestStatus


@pytest.fixture(autouse=True)
def _clean_hooks():
    hooks.clear()
    yield
    hooks.clear()


# ----------------------------------------------------------------- registry


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        hooks.register("scheduler.no_such_hook", lambda *a: None)


def test_every_hook_has_a_known_kind():
    for name, (kind, doc) in hooks.HOOKS.items():
        assert kind in (hooks.NOTIFY, hooks.FIRST, hooks.FOLD, hooks.CONTEXT), name
        assert doc


def test_active_register_unregister():
    fn = lambda *a: None  # noqa: E731
    assert not hooks.active("scheduler.init")
    assert hooks.register("scheduler.init", fn) is fn
    assert hooks.active("scheduler.init")
    hooks.unregister("scheduler.init", fn)
    assert not hooks.active("scheduler.init")
    hooks.unregister("scheduler.init", fn)  # no-op


def test_notify_runs_all_and_swallows_errors():
    seen = []

    def bad(x):
        raise RuntimeError("observer failure")

    hooks.register("scheduler.init", lambda x: seen.append(("a", x)))
    hooks.register("scheduler.init", bad)
    hooks.register("scheduler.init", lambda x: seen.append(("b", x)))
    hooks.notify("scheduler.init", 7)
    assert seen == [("a", 7), ("b", 7)]


def test_first_returns_first_non_none_in_order():
    assert hooks.first("scheduler.admit", None, None) is None
    hooks.register("scheduler.admit", lambda s, r: None)
    hooks.register("scheduler.admit", lambda s, r: False)
    hooks.register("scheduler.admit", lambda s, r: True)
    assert hooks.first("scheduler.admit", None, None) is False


def test_fold_chains_values():
    assert hooks.fold("worker.available_kv_memory", 100, None) == 100
    hooks.register("worker.available_kv_memory", lambda v, w: v + 10)
    hooks.register("worker.available_kv_memory", lambda v, w: v * 2)
    assert hooks.fold("worker.available_kv_memory", 100, None) == 220


def test_context_enters_in_order_and_exits_reversed():
    log = []

    @contextlib.contextmanager
    def cm(tag):
        log.append("enter " + tag)
        try:
            yield
        finally:
            log.append("exit " + tag)

    hooks.register("model_runner.execute_model", lambda r: cm("a"))
    hooks.register("model_runner.execute_model", lambda r: None)
    hooks.register("model_runner.execute_model", lambda r: cm("b"))
    with hooks.context("model_runner.execute_model", object()):
        log.append("body")
    assert log == ["enter a", "enter b", "body", "exit b", "exit a"]


def test_around_and_after_decorators():
    calls = []

    class Thing:
        @hooks.around("kv_cache_manager.allocate_slots", lambda s, a, k: (s, a[0]))
        def allocate_slots(self, request, n=1):
            calls.append(("call", request, n))
            return n

        @hooks.after("worker.available_kv_memory")
        def determine_available_memory(self):
            return 1000

        @hooks.after("worker.warmed_up")
        def compile_or_warm_up_model(self):
            return "times"

    t = Thing()
    # nothing registered: plain calls
    assert t.allocate_slots("r", n=3) == 3
    assert t.determine_available_memory() == 1000
    assert t.compile_or_warm_up_model() == "times"

    @contextlib.contextmanager
    def ctx(mgr, req):
        calls.append(("enter", req))
        yield
        calls.append(("exit", req))

    hooks.register("kv_cache_manager.allocate_slots", ctx)
    hooks.register("worker.available_kv_memory", lambda v, w: v - 1)
    warmed = []
    hooks.register("worker.warmed_up", warmed.append)
    assert t.allocate_slots("q", n=2) == 2
    assert calls[-3:] == [("enter", "q"), ("call", "q", 2), ("exit", "q")]
    assert t.determine_available_memory() == 999
    assert t.compile_or_warm_up_model() == "times"
    assert warmed == [t]


# ----------------------------------------------------------------- block pool


def _pool(n=10, caching=True):
    return BlockPool(num_gpu_blocks=n, enable_caching=caching, hash_block_size=16)


def test_get_new_blocks_default_unchanged():
    a, b = _pool(), _pool()
    hooks.register("block_pool.touch", lambda p, blocks: None)  # unrelated hook
    ids_a = [blk.block_id for blk in a.get_new_blocks(3)]
    hooks.clear()
    ids_b = [blk.block_id for blk in b.get_new_blocks(3)]
    assert ids_a == ids_b == [1, 2, 3]


def test_get_new_blocks_override_places_blocks():
    pool = _pool()

    def top_first(p, n):
        picked = []
        cur = p.free_block_queue.fake_free_list_tail.prev_free_block
        while len(picked) < n:
            picked.append(cur)
            cur = cur.prev_free_block
        for blk in picked:
            p.free_block_queue.remove(blk)
            p._maybe_evict_cached_block(blk)
            blk.ref_cnt += 1
        return picked

    hooks.register("block_pool.get_new_blocks", top_first)
    got = pool.get_new_blocks(2)
    assert [b.block_id for b in got] == [9, 8]
    assert all(b.ref_cnt == 1 for b in got)
    assert pool.get_num_free_blocks() == 9 - 2
    # None from the hook = default placement
    hooks.clear()
    hooks.register("block_pool.get_new_blocks", lambda p, n: None)
    assert [b.block_id for b in pool.get_new_blocks(1)] == [1]


def test_block_pool_observers():
    pool = _pool()
    seen = []
    hooks.register("block_pool.touch", lambda p, blocks: seen.append(len(blocks)))
    hooks.register(
        "block_pool.get_cached_block", lambda p, h: seen.append(("lookup", h))
    )
    blocks = pool.get_new_blocks(2)
    pool.touch(blocks)
    assert pool.get_cached_block(b"x" * 32, [0]) is None
    assert seen == [2, ("lookup", b"x" * 32)]


def test_get_usage_leaves_lent_blocks_out():
    pool = _pool(n=11)  # 10 usable + null block
    assert pool.num_lent_blocks == 0
    pool.get_new_blocks(5)
    assert pool.get_usage() == pytest.approx(0.5)
    # lend 4 of the 5 free blocks: they leave the free queue and the total
    lent = pool.get_new_blocks(4)
    pool.num_lent_blocks = len(lent)
    assert pool.get_usage() == pytest.approx(1.0 - 1 / 6)
    pool.free_blocks(lent)
    pool.num_lent_blocks = 0
    assert pool.get_usage() == pytest.approx(0.5)


# ----------------------------------------------------------------- scheduler


@pytest.fixture
def tiny_model(tmp_path, monkeypatch):
    """A local OPT-style config (no download; tokenizer init skipped). On a
    machine without an accelerator the platform has no device type; the
    scheduler only needs one to build its config."""
    from vllm.platforms import current_platform

    if not current_platform.device_type:
        monkeypatch.setattr(current_platform, "device_type", "cpu")
    cfg = {
        "architectures": ["OPTForCausalLM"],
        "model_type": "opt",
        "hidden_size": 64,
        "ffn_dim": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "max_position_embeddings": 2048,
        "vocab_size": 1000,
        "word_embed_proj_dim": 64,
        "do_layer_norm_before": True,
        "torch_dtype": "float16",
        "eos_token_id": 2,
        "bos_token_id": 2,
        "pad_token_id": 1,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


def _scheduler(model):
    from .utils import create_scheduler

    return create_scheduler(model=model, skip_tokenizer_init=True, num_blocks=100)


def test_scheduler_hooks_and_admit(tiny_model):
    from .utils import create_requests

    seen = []
    hooks.register("scheduler.init", lambda s: seen.append("init"))
    sched = _scheduler(tiny_model)
    assert seen == ["init"]

    held = {"r1"}

    def admit(s, request):
        return False if request.request_id in held else None

    hooks.register("scheduler.before_schedule", lambda s: seen.append("before"))
    hooks.register(
        "scheduler.after_schedule",
        lambda s, out: seen.append(("after", sorted(out.num_scheduled_tokens))),
    )
    hooks.register("scheduler.admit", admit)

    reqs = create_requests(num_requests=3, num_tokens=10, req_ids=["r0", "r1", "r2"])
    for r in reqs:
        sched.add_request(r)
    out = sched.schedule()
    assert sorted(out.num_scheduled_tokens) == ["r0", "r2"]
    assert seen[1:] == ["before", ("after", ["r0", "r2"])]
    by_id = {r.request_id: r for r in reqs}
    assert by_id["r1"].status == RequestStatus.WAITING
    assert by_id["r1"] in list(sched.waiting) + list(sched.skipped_waiting)

    held.clear()
    out = sched.schedule()
    assert "r1" in out.num_scheduled_tokens


def test_scheduler_without_hooks_unchanged(tiny_model):
    from .utils import create_requests

    a, b = _scheduler(tiny_model), _scheduler(tiny_model)
    hooks.register("scheduler.admit", lambda s, r: None)  # default answer
    for r in create_requests(num_requests=4, num_tokens=10):
        a.add_request(r)
    out_a = a.schedule()
    hooks.clear()
    for r in create_requests(num_requests=4, num_tokens=10):
        b.add_request(r)
    out_b = b.schedule()
    assert out_a.num_scheduled_tokens == out_b.num_scheduled_tokens
