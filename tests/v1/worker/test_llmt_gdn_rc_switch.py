# SPDX-License-Identifier: Apache-2.0
"""GDN-RECOMPUTE runner switch follows the EXL3 plugin's GDN_RC["on"] (CPU only, no GPU needed)."""
import sys
import types

from vllm.v1.worker import gpu_model_runner as gmr


def _with_plugin(monkeypatch, on):
    pkg = types.ModuleType("exl3cuda")
    mod = types.ModuleType("exl3cuda.vllm_plugin")
    mod.GDN_RC = {"on": on}
    pkg.vllm_plugin = mod
    monkeypatch.setitem(sys.modules, "exl3cuda", pkg)
    monkeypatch.setitem(sys.modules, "exl3cuda.vllm_plugin", mod)
    gmr._gdn_rc_on.cache_clear()


def test_follows_plugin(monkeypatch):
    for on in (True, False):
        _with_plugin(monkeypatch, on)
        assert gmr._gdn_rc_on() is on
    gmr._gdn_rc_on.cache_clear()


def test_off_without_plugin(monkeypatch):
    monkeypatch.setitem(sys.modules, "exl3cuda", None)      # import fails
    monkeypatch.setitem(sys.modules, "exl3cuda.vllm_plugin", None)
    gmr._gdn_rc_on.cache_clear()
    assert gmr._gdn_rc_on() is False
    gmr._gdn_rc_on.cache_clear()
