# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named extension hooks for memory-management plugins.

A plugin (for example one loaded through the ``vllm.general_plugins`` entry
point, which runs in every vLLM process) registers callables under the names in
:data:`HOOKS`. vLLM core calls them at fixed points in the scheduler, the block
pool, the KV cache manager, the model runner and the worker. With nothing
registered every call site costs one dict lookup and behaviour is unchanged.

Kinds:

* ``notify``: observers; every callable runs, results are ignored, exceptions
  are logged and swallowed (an observer must not take the engine down).
* ``first``: overrides; callables run in registration order and the first
  non-``None`` result wins. ``None`` from all = the default behaviour.
  Exceptions propagate.
* ``fold``: transforms; ``value = fn(value, *args)`` for each callable.
* ``context``: each callable returns a context manager (or ``None``); the call
  site runs inside all of them (entered in registration order).

Example::

    from vllm.v1 import extension_hooks as hooks

    def register():  # vllm.general_plugins entry point
        hooks.register("scheduler.before_schedule", lambda sched: ...)
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

NOTIFY = "notify"
FIRST = "first"
FOLD = "fold"
CONTEXT = "context"

# name -> (kind, description)
HOOKS: dict[str, tuple[str, str]] = {
    "scheduler.init": (
        NOTIFY,
        "(scheduler): end of Scheduler.__init__.",
    ),
    "scheduler.before_schedule": (
        NOTIFY,
        "(scheduler): start of Scheduler.schedule, before any request is "
        "scheduled in this step.",
    ),
    "scheduler.after_schedule": (
        NOTIFY,
        "(scheduler, scheduler_output): end of Scheduler.schedule.",
    ),
    "scheduler.admit": (
        FIRST,
        "(scheduler, request) -> bool | None: called for each waiting request "
        "before it is considered in this step. False keeps it waiting (it goes "
        "to skipped_waiting, like other requests that cannot be scheduled "
        "yet); None or True = default. Extension point for admission hold and "
        "priority policies.",
    ),
    "block_pool.get_new_blocks": (
        FIRST,
        "(pool, num_blocks) -> list[KVCacheBlock] | None: start of "
        "BlockPool.get_new_blocks. A result must be exactly num_blocks blocks "
        "taken from the free queue, ref_cnt incremented and cached hashes "
        "evicted, as the default does. None = default. Placement policy.",
    ),
    "block_pool.cache_full_blocks": (
        NOTIFY,
        "(pool, request, num_cached_blocks, num_full_blocks): end of "
        "BlockPool.cache_full_blocks.",
    ),
    "block_pool.touch": (
        NOTIFY,
        "(pool, blocks): start of BlockPool.touch (prefix-cache hit).",
    ),
    "block_pool.get_cached_block": (
        NOTIFY,
        "(pool, block_hash): start of BlockPool.get_cached_block.",
    ),
    "kv_cache_manager.allocate_slots": (
        CONTEXT,
        "(manager, request): around KVCacheManager.allocate_slots.",
    ),
    "model_runner.kv_caches_initialized": (
        NOTIFY,
        "(runner, kv_caches, kv_cache_config): end of "
        "GPUModelRunner.initialize_kv_cache_tensors.",
    ),
    "model_runner.execute_model": (
        CONTEXT,
        "(runner): around GPUModelRunner.execute_model.",
    ),
    "model_runner.model_forward": (
        CONTEXT,
        "(runner, args, kwargs): around GPUModelRunner._model_forward "
        "(kwargs is the dict passed to the forward; read-only).",
    ),
    "model_runner.propose_draft": (
        CONTEXT,
        "(runner): around GPUModelRunner.propose_draft_token_ids.",
    ),
    "worker.available_kv_memory": (
        FOLD,
        "(bytes, worker) -> bytes: result of Worker.determine_available_memory "
        "(the KV cache budget).",
    ),
    "worker.warmed_up": (
        NOTIFY,
        "(worker): end of Worker.compile_or_warm_up_model.",
    ),
}

_registry: dict[str, list[Callable[..., Any]]] = {}


def _check(name: str) -> None:
    if name not in HOOKS:
        raise ValueError(
            f"unknown extension hook {name!r}; known: {', '.join(sorted(HOOKS))}"
        )


def register(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register ``fn`` under ``name``; returns ``fn`` (usable as decorator
    via ``functools.partial(register, name)``)."""
    _check(name)
    _registry.setdefault(name, []).append(fn)
    return fn


def unregister(name: str, fn: Callable[..., Any]) -> None:
    _check(name)
    fns = _registry.get(name)
    if fns and fn in fns:
        fns.remove(fn)
        if not fns:
            del _registry[name]


def clear() -> None:
    """Remove every registration (tests)."""
    _registry.clear()


def active(name: str) -> bool:
    return name in _registry


def notify(name: str, *args: Any) -> None:
    for fn in _registry.get(name, ()):
        try:
            fn(*args)
        except Exception:
            logger.exception("extension hook %s (%r) failed", name, fn)


def first(name: str, *args: Any) -> Any:
    for fn in _registry.get(name, ()):
        r = fn(*args)
        if r is not None:
            return r
    return None


def fold(name: str, value: Any, *args: Any) -> Any:
    for fn in _registry.get(name, ()):
        value = fn(value, *args)
    return value


def context(name: str, *args: Any) -> contextlib.AbstractContextManager:
    fns = _registry.get(name)
    if not fns:
        return contextlib.nullcontext()
    stack = contextlib.ExitStack()
    with stack:
        for fn in fns:
            cm = fn(*args)
            if cm is not None:
                stack.enter_context(cm)
        return stack.pop_all()


def around(
    name: str,
    hook_args: Callable[[Any, tuple, dict], tuple] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Method decorator for a ``context`` hook: runs the method inside
    :func:`context` (``name``). ``hook_args(self, args, kwargs)`` gives the
    hook arguments (default ``(self,)``). With nothing registered the wrapper
    calls the method directly."""
    _check(name)

    def deco(method: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if name not in _registry:
                return method(self, *args, **kwargs)
            hargs = hook_args(self, args, kwargs) if hook_args else (self,)
            with context(name, *hargs):
                return method(self, *args, **kwargs)

        return wrapper

    return deco


def after(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Method decorator for a ``notify`` or ``fold`` hook called with the
    method's result: ``notify(name, self)`` after the call, or
    ``result = fold(name, result, self)``. With nothing registered the wrapper
    calls the method directly."""
    _check(name)
    kind = HOOKS[name][0]
    assert kind in (NOTIFY, FOLD), name

    def deco(method: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            if name not in _registry:
                return result
            if kind == FOLD:
                return fold(name, result, self)
            notify(name, self)
            return result

        return wrapper

    return deco
