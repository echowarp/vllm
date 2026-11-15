# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Defaults of this branch's EXL3_* engine switches.

An unset switch takes the value of the reference build (Qwen3.8-27B EXL3 on an Intel Arc Pro B70 with the exl3xpu plugin, int4 KV, MTP k = 7; the plugin's
models/<id>/model.yaml) only while the EXL3 plugin serves the model: the
model's quantization is "exl3" (registered by exl3xpu) on an XPU. Any other model, or this branch without
the plugin, keeps upstream behaviour. A switch that is set (also to an empty string) always wins.

activate(vllm_config) decides once per process; the Scheduler and GPUModelRunner constructors call it first. Module
constants made with flag() are re-evaluated then; env() reads the current state on every call.
"""
import os
import sys
from collections.abc import Callable
from typing import Any

_STATE = {"active": False, "decided": False}
_REG: list[tuple[str, str, str, str, str, Callable[[str], Any]]] = []


def _ref_hw() -> bool:
    try:
        from vllm.platforms import current_platform
        return bool(current_platform.is_xpu())
    except Exception:  # noqa: BLE001
        return False


def active() -> bool:
    return _STATE["active"]


def env(name: str, ref: str, other: str) -> str:
    """os.environ[name] if set, else ref while active() and other otherwise."""
    v = os.environ.get(name)
    if v is not None:
        return v
    return ref if _STATE["active"] else other


def flag(module: str, attr: str, name: str, ref: str, other: str, conv: Callable[[str], Any] = str) -> Any:
    """A module constant: registered so activate() can set module.attr = conv(env(name, ref, other))."""
    _REG.append((module, attr, name, ref, other, conv))
    return conv(env(name, ref, other))


def activate(vllm_config: Any) -> bool:
    mc = getattr(vllm_config, "model_config", None)
    act = getattr(mc, "quantization", None) == "exl3" and _ref_hw()
    if act != _STATE["active"] or not _STATE["decided"]:
        _STATE["active"], _STATE["decided"] = act, True
        for module, attr, name, ref, other, conv in _REG:
            m = sys.modules.get(module)
            if m is not None:
                setattr(m, attr, conv(env(name, ref, other)))
    return act
