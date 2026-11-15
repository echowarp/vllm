# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Defaults of this branch's EXL3_* engine switches.

An unset switch takes the value of the reference build (Qwen3.8-27B EXL3 on an Intel Arc Pro B70 with the exl3xpu
plugin, int4 KV, MTP k = 7; the plugin's models/<id>/model.yaml) when vLLM runs on an XPU, and upstream's behaviour
otherwise. A switch that is set (also to an empty string) always wins.
"""
import functools
import os


@functools.lru_cache(maxsize=1)
def on_xpu() -> bool:
    try:
        from vllm.platforms import current_platform
        return bool(current_platform.is_xpu())
    except Exception:  # noqa: BLE001
        return False


def env(name: str, ref: str, other: str) -> str:
    """os.environ[name] if set, else ref on an XPU and other elsewhere."""
    v = os.environ.get(name)
    if v is not None:
        return v
    return ref if on_xpu() else other
