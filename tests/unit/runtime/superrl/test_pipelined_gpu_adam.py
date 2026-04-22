# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
"""Unit test: SuperRLPipelinedGPUAdam matches reference AdamW on CPU.

Uses a fake DeepNVMe handle (in-memory file store) so the test runs in any
environment.
"""
import os

import pytest
import torch

from deepspeed.runtime.superrl.io.nvme_engine import CoalescedNVMeEngine
from deepspeed.runtime.superrl.io.pipelined_gpu_adam import SuperRLPipelinedGPUAdam


class _FakeHandle:
    """Shared-storage in-memory DeepNVMe stand-in.

    The pipelined Adam now creates a separate write engine so reads and
    writes overlap; the test only works if both engines see the same
    backing store, so we keep ``files`` at class scope.
    """

    files: dict = {}

    def __init__(self):
        self._pending = []

    def async_pwrite(self, tensor, path):
        self._pending.append(("w", tensor, path))

    def async_pread(self, tensor, path):
        self._pending.append(("r", tensor, path))

    def wait(self):
        for op, tensor, path in self._pending:
            if op == "w":
                _FakeHandle.files[path] = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            else:
                raw = _FakeHandle.files.get(path, b"")
                if not raw:
                    continue
                nbytes = tensor.numel() * tensor.element_size()
                buf = torch.frombuffer(bytearray(raw[:nbytes]), dtype=tensor.dtype)
                tensor.view(-1)[:buf.numel()].copy_(buf)
        self._pending.clear()


@pytest.fixture(autouse=True)
def _clear_fake_files():
    _FakeHandle.files.clear()
    yield
    _FakeHandle.files.clear()


@pytest.fixture(autouse=True)
def fake_factory(monkeypatch):
    def _fake_resolve(self):
        return self.HANDLE_AIO, _FakeHandle
    monkeypatch.setattr(CoalescedNVMeEngine, "_resolve_handle_factory", _fake_resolve)
    yield


def make_superrl_adam(params, swap_dir, lr=1e-3, wd=0.01):
    engine = CoalescedNVMeEngine(
        nvme_devices=[swap_dir],
        block_size=1024, queue_depth=4,
        intra_op_parallelism=1, single_submit=False,
        overlap_events=False, use_gds=False,
    )
    return SuperRLPipelinedGPUAdam(
        params,
        lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd,
        nvme_engine=engine,
        swap_folder=swap_dir,
        gpu_device=torch.device("cpu"),
        chunk_bytes=4 * 1024,  # small chunks so the ring is exercised
        ring_depth=2,
    )


def test_adam_matches_reference_within_tolerance(tmp_path):
    torch.manual_seed(42)
    n = 128
    ref_p = torch.randn(n, requires_grad=True)
    ref_opt = torch.optim.AdamW([ref_p], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    srl_p = ref_p.detach().clone().requires_grad_(True)
    srl_opt = make_superrl_adam([srl_p], str(tmp_path / "swap"))

    for _ in range(10):
        grad = torch.randn(n)
        ref_p.grad = grad.clone()
        srl_p.grad = grad.clone()
        ref_opt.step()
        srl_opt.step()

    max_diff = (ref_p.detach() - srl_p.detach()).abs().max().item()
    assert max_diff < 1e-4, f"max param diff {max_diff} exceeds threshold"


def test_step_with_no_grad_is_noop(tmp_path):
    p = torch.randn(16, requires_grad=True)
    before = p.detach().clone()
    opt = make_superrl_adam([p], str(tmp_path / "swap"))
    opt.step()
    assert torch.allclose(p.detach(), before)


def test_stats_returned(tmp_path):
    p = torch.randn(8, requires_grad=True)
    p.grad = torch.randn(8)
    opt = make_superrl_adam([p], str(tmp_path / "swap"))
    opt.step()
    stats = opt.stats()
    assert "superrl_io/bytes_written" in stats
    assert "superrl_io/optimizer_chunks" in stats
    assert stats["superrl_io/optimizer_chunks"] >= 1


def test_chunking_across_many_params(tmp_path):
    """Many small params should still fit a single step without hanging."""
    torch.manual_seed(0)
    params = [torch.randn(7, requires_grad=True) for _ in range(8)]
    for p in params:
        p.grad = torch.randn_like(p)
    opt = make_superrl_adam(params, str(tmp_path / "swap"))
    opt.step()
    # Ring should have been reused since chunk_bytes is small.
    assert opt._max_chunk_numel >= 7
