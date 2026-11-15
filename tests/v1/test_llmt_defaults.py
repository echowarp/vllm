# SPDX-License-Identifier: Apache-2.0
"""This branch's EXL3_* switch defaults (CPU only): reference values on an XPU, upstream elsewhere, set values win;
the GDN-RECOMPUTE runner switch follows the exl3xpu plugin's GDN_RC["on"]."""
import sys
import types

from vllm.v1 import llmt_defaults
from vllm.v1.worker import gpu_model_runner as gmr


def test_env_rule(monkeypatch):
    monkeypatch.delenv("EXL3_T_X", raising=False)
    for xpu, want in ((True, "ref"), (False, "off")):
        monkeypatch.setattr(llmt_defaults, "on_xpu", lambda xpu=xpu: xpu)
        assert llmt_defaults.env("EXL3_T_X", "ref", "off") == want
    monkeypatch.setenv("EXL3_T_X", "")
    assert llmt_defaults.env("EXL3_T_X", "ref", "off") == ""


def test_gdn_rc_follows_plugin(monkeypatch):
    for on in (True, False):
        pkg, mod = types.ModuleType("exl3xpu"), types.ModuleType("exl3xpu.vllm_plugin")
        mod.GDN_RC = {"on": on}
        pkg.vllm_plugin = mod
        monkeypatch.setitem(sys.modules, "exl3xpu", pkg)
        monkeypatch.setitem(sys.modules, "exl3xpu.vllm_plugin", mod)
        gmr._gdn_rc_on.cache_clear()
        assert gmr._gdn_rc_on() is on
    monkeypatch.setitem(sys.modules, "exl3xpu", None)
    monkeypatch.setitem(sys.modules, "exl3xpu.vllm_plugin", None)
    gmr._gdn_rc_on.cache_clear()
    assert gmr._gdn_rc_on() is False
    gmr._gdn_rc_on.cache_clear()
