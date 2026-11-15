# SPDX-License-Identifier: Apache-2.0
"""This branch's EXL3_* switch defaults (CPU only): reference values only while the EXL3 plugin serves the model,
upstream otherwise, set values win; the GDN-RECOMPUTE runner switch follows the exl3xpu plugin."""
import sys
import types

import pytest

from vllm.v1 import llmt_defaults as L
from vllm.v1.core.sched import scheduler as sched
from vllm.v1.worker import gpu_model_runner as gmr


def _cfg(quant):
    return types.SimpleNamespace(model_config=types.SimpleNamespace(quantization=quant))


@pytest.fixture
def hw(monkeypatch):
    monkeypatch.setattr(L, "_ref_hw", lambda: True)
    yield
    L.activate(_cfg(None))            # back to upstream defaults for the other tests


def test_inactive_without_exl3(hw, monkeypatch):
    monkeypatch.delenv("EXL3_EAGLE_KEEP", raising=False)
    for q in (None, "fp8", "gptq"):
        assert L.activate(_cfg(q)) is False
        assert sched._EAGLE_KEEP is False and gmr._ALIGN_NOSYNC is False
        assert L.env("EXL3_T_X", "ref", "off") == "off"


def test_active_with_exl3_model(hw, monkeypatch):
    monkeypatch.delenv("EXL3_EAGLE_KEEP", raising=False)
    monkeypatch.delenv("EXL3_ALIGN_NOSYNC", raising=False)
    monkeypatch.delenv("EXL3_T_X", raising=False)
    assert L.activate(_cfg("exl3")) is True
    assert sched._EAGLE_KEEP is True and sched._EAGLE_KEEP_MODE == "2" and gmr._ALIGN_NOSYNC is True
    assert L.env("EXL3_T_X", "ref", "off") == "ref"
    monkeypatch.setenv("EXL3_T_X", "")
    assert L.env("EXL3_T_X", "ref", "off") == ""


def test_exl3_on_other_hardware(monkeypatch):
    monkeypatch.setattr(L, "_ref_hw", lambda: False)
    assert L.activate(_cfg("exl3")) is False


def _plugin(monkeypatch, on):
    pkg, mod = types.ModuleType("exl3xpu"), types.ModuleType("exl3xpu.vllm_plugin")
    mod.GDN_RC = {"on": on}
    pkg.vllm_plugin = mod
    monkeypatch.setitem(sys.modules, "exl3xpu", pkg)
    monkeypatch.setitem(sys.modules, "exl3xpu.vllm_plugin", mod)
    gmr._gdn_rc_on.cache_clear()


def test_gdn_rc_follows_plugin(hw, monkeypatch):
    monkeypatch.delenv("EXL3_GDN_RECOMPUTE", raising=False)
    L.activate(_cfg("exl3"))
    for on in (True, False):
        _plugin(monkeypatch, on)
        assert gmr._gdn_rc_on() is on
    L.activate(_cfg("fp8"))                       # not the plugin's model: off unless forced
    _plugin(monkeypatch, True)
    assert gmr._gdn_rc_on() is False
    monkeypatch.setenv("EXL3_GDN_RECOMPUTE", "1")
    gmr._gdn_rc_on.cache_clear()
    assert gmr._gdn_rc_on() is True
    monkeypatch.setitem(sys.modules, "exl3xpu", None)
    monkeypatch.setitem(sys.modules, "exl3xpu.vllm_plugin", None)
    gmr._gdn_rc_on.cache_clear()
    assert gmr._gdn_rc_on() is False
    gmr._gdn_rc_on.cache_clear()
