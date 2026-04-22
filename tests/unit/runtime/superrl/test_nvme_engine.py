# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CoalescedNVMeEngine - runnable without GH200 or libaio.

These tests inject a fake handle factory in place of the DeepNVMe builder,
so they exercise the SuperRL contributions (coalescing planner, per-device
router, stats) without requiring DeepNVMe to be compiled.
"""
import os
import threading

import pytest
import torch

from deepspeed.runtime.superrl.io import nvme_engine as ne_mod
from deepspeed.runtime.superrl.io.nvme_engine import CoalescedNVMeEngine, IORequest


class _FakeHandle:
    """In-memory stand-in for an aio_handle. Records every (op, tensor, path)."""

    def __init__(self):
        self.ops = []  # list of (op, ptr, nbytes, path)
        self.files = {}  # path -> bytes
        self._pending = []
        self._lock = threading.Lock()

    def _record(self, op, tensor, path):
        with self._lock:
            self._pending.append((op, tensor, path))

    def async_pread(self, tensor, path):
        self._record("read", tensor, path)

    def async_pwrite(self, tensor, path):
        self._record("write", tensor, path)

    def wait(self):
        with self._lock:
            for op, tensor, path in self._pending:
                if op == "write":
                    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                    self.files[path] = raw
                else:
                    raw = self.files.get(path, b"")
                    nbytes = tensor.numel() * tensor.element_size()
                    if not raw:
                        continue
                    buf = torch.frombuffer(bytearray(raw[:nbytes]), dtype=tensor.dtype)
                    tensor.view(-1)[:buf.numel()].copy_(buf)
            self._pending.clear()


@pytest.fixture(autouse=True)
def fake_factory(monkeypatch):
    """Replace the DeepNVMe handle factory so we don't need libaio."""
    def _fake_resolve(self):
        return self.HANDLE_AIO, _FakeHandle
    monkeypatch.setattr(CoalescedNVMeEngine, "_resolve_handle_factory", _fake_resolve)
    yield


def make_engine(tmpdir, n_devices=1):
    devices = [str(tmpdir / f"d{i}") for i in range(n_devices)] if n_devices > 1 else [str(tmpdir)]
    for d in devices:
        os.makedirs(d, exist_ok=True)
    return CoalescedNVMeEngine(
        nvme_devices=devices,
        block_size=1024,
        queue_depth=8,
        intra_op_parallelism=1,
        single_submit=False,
        overlap_events=True,
        use_gds=False,
    )


# --- Coalescing planner -----------------------------------------------------


def test_coalescing_merges_contiguous_views(tmp_path):
    engine = make_engine(tmp_path)
    base = torch.zeros(100, dtype=torch.float32)
    elem = base.element_size()
    reqs = [
        IORequest(buffer=base[i * 25:(i + 1) * 25],
                  path=str(tmp_path / "p.bin"),
                  offset=i * 25 * elem,
                  group="g")
        for i in range(4)
    ]
    merged = engine._coalesce(reqs)
    assert len(merged) == 1
    assert merged[0].buffer.numel() == 100


def test_coalescing_does_not_merge_distinct_paths(tmp_path):
    engine = make_engine(tmp_path)
    base = torch.zeros(50, dtype=torch.float32)
    elem = base.element_size()
    reqs = [
        IORequest(buffer=base[:25], path=str(tmp_path / "a.bin"), offset=0, group="g"),
        IORequest(buffer=base[25:], path=str(tmp_path / "b.bin"), offset=0, group="g"),
    ]
    merged = engine._coalesce(reqs)
    assert len(merged) == 2


def test_coalescing_does_not_merge_distinct_groups(tmp_path):
    engine = make_engine(tmp_path)
    base = torch.zeros(50, dtype=torch.float32)
    elem = base.element_size()
    reqs = [
        IORequest(buffer=base[:25], path=str(tmp_path / "p.bin"), offset=0, group="m"),
        IORequest(buffer=base[25:], path=str(tmp_path / "p.bin"), offset=25 * elem, group="v"),
    ]
    merged = engine._coalesce(reqs)
    assert len(merged) == 2


def test_coalescing_skips_non_contiguous_storage(tmp_path):
    engine = make_engine(tmp_path)
    a = torch.zeros(25, dtype=torch.float32)
    b = torch.zeros(25, dtype=torch.float32)
    reqs = [
        IORequest(buffer=a, path=str(tmp_path / "p.bin"), offset=0, group="g"),
        IORequest(buffer=b, path=str(tmp_path / "p.bin"), offset=25 * a.element_size(), group="g"),
    ]
    merged = engine._coalesce(reqs)
    assert len(merged) == 2  # different storages -> no merge


# --- Per-device routing ------------------------------------------------------


def test_stripe_routing_single(tmp_path):
    engine = make_engine(tmp_path)
    assert engine._stripe_for("/anywhere/x.bin") == 0


def test_stripe_routing_multi_is_deterministic(tmp_path):
    engine = make_engine(tmp_path, n_devices=2)
    s1 = engine._stripe_for("/anywhere/foo.bin")
    s2 = engine._stripe_for("/anywhere/foo.bin")
    s3 = engine._stripe_for("/elsewhere/bar.bin")
    assert s1 == s2
    assert 0 <= s1 < 2 and 0 <= s3 < 2


# --- Round-trip submit + wait -----------------------------------------------


def test_submit_writes_then_reads_round_trip(tmp_path):
    engine = make_engine(tmp_path)
    data = torch.randn(64, dtype=torch.float32)
    path = str(tmp_path / "param.bin")
    engine.submit_writes([IORequest(buffer=data, path=path, offset=0)])
    engine.wait_all()
    result = torch.zeros(64, dtype=torch.float32)
    engine.submit_reads([IORequest(buffer=result, path=path, offset=0)])
    engine.wait_all()
    assert torch.allclose(data, result, atol=1e-6)


# --- Stats ------------------------------------------------------------------


def test_stats_keys_present(tmp_path):
    engine = make_engine(tmp_path)
    stats = engine.stats()
    for key in (
        "superrl_io/handle_kind",
        "superrl_io/stripes",
        "superrl_io/bytes_read",
        "superrl_io/bytes_written",
        "superrl_io/raw_requests",
        "superrl_io/submitted_requests",
        "superrl_io/coalescing_ratio",
    ):
        assert key in stats


def test_coalescing_ratio_reflects_planner(tmp_path):
    engine = make_engine(tmp_path)
    base = torch.zeros(100, dtype=torch.float32)
    elem = base.element_size()
    reqs = [
        IORequest(buffer=base[i * 25:(i + 1) * 25],
                  path=str(tmp_path / "p.bin"),
                  offset=i * 25 * elem,
                  group="g")
        for i in range(4)
    ]
    engine.submit_writes(reqs)
    engine.wait_all()
    stats = engine.stats()
    # 4 raw requests folded into 1 submitted -> ratio = 0.75
    assert stats["superrl_io/raw_requests"] == 4
    assert stats["superrl_io/submitted_requests"] == 1
    assert stats["superrl_io/coalescing_ratio"] == pytest.approx(0.75)
