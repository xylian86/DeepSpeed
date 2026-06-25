# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import pytest
import torch

from deepspeed.runtime.swap_tensor import utils as swap_utils
from deepspeed.runtime.swap_tensor import optimizer_utils as optimizer_swap_utils
from deepspeed.runtime.swap_tensor.optimizer_utils import OptimizerSwapper, split_swap_buffer_counts


class _FakeAccelerator:

    def pin_memory(self, tensor, align_bytes=0):
        return tensor


class _FakeAIOHandle:

    def __init__(self):
        self.writes = []
        self.pending_writes = 0
        self.wait_counts = []

    def async_pwrite(self, buffer, path, offset):
        self.writes.append((buffer.clone(), path, offset))
        self.pending_writes += 1
        return 0

    def wait(self):
        wait_count = self.pending_writes
        self.wait_counts.append(wait_count)
        self.pending_writes = 0
        return wait_count


def _patch_swap_buffer_manager_deps(monkeypatch):
    monkeypatch.setattr(swap_utils, "get_accelerator", lambda: _FakeAccelerator())
    monkeypatch.setattr(swap_utils.dist, "get_rank", lambda: 1)


def test_swap_buffer_manager_uses_empty_for_pool(monkeypatch):
    _patch_swap_buffer_manager_deps(monkeypatch)

    real_empty = swap_utils.torch.empty
    empty_calls = []

    def tracked_empty(*args, **kwargs):
        empty_calls.append((args, kwargs))
        return real_empty(*args, **kwargs)

    def fail_zeros(*args, **kwargs):
        raise AssertionError("SwapBufferManager should not zero-fill staging buffers")

    monkeypatch.setattr(swap_utils.torch, "empty", tracked_empty)
    monkeypatch.setattr(swap_utils.torch, "zeros", fail_zeros)

    manager = swap_utils.SwapBufferManager(num_elems=8, count=2, dtype=torch.float32)

    assert len(empty_calls) == 2
    assert manager.status()["total_bytes"] == 8 * 2 * torch.tensor([], dtype=torch.float32).element_size()
    assert manager.status()["buffer_bytes"] == 8 * torch.tensor([], dtype=torch.float32).element_size()
    assert manager.status()["pin_memory_time_sec"] >= 0


def test_swap_buffer_manager_reports_allocation_state(monkeypatch):
    _patch_swap_buffer_manager_deps(monkeypatch)

    manager = swap_utils.SwapBufferManager(num_elems=8, count=2, dtype=torch.float32)

    buffers = manager.allocate(num_elems=4, count=1, dtype=torch.float32)
    assert buffers is not None
    assert manager.status()["free_buffer_count"] == 1
    assert manager.status()["used_buffer_count"] == 1
    assert manager.status()["max_allocated_buffers"] == 1
    assert manager.status()["max_allocated_bytes"] == manager.status()["buffer_bytes"]

    assert manager.allocate(num_elems=4, count=2, dtype=torch.float32) is None
    status = manager.status()
    assert status["num_failed_allocations"] == 1
    assert status["max_requested_count"] == 2
    assert status["max_requested_bytes"] == 4 * 2 * torch.tensor([], dtype=torch.float32).element_size()

    message = manager.allocation_failure_message(requested_num_elems=4,
                                                 requested_count=2,
                                                 owner="unit test")
    assert "unit test" in message
    assert "free_buffer_count=1" in message
    assert "used_buffer_count=1" in message
    assert "buffer_count=2" in message

    manager.free(buffers)
    assert manager.status()["free_buffer_count"] == 2


def test_swap_buffer_manager_lazy_slots_allocate_on_demand(monkeypatch):
    _patch_swap_buffer_manager_deps(monkeypatch)

    real_empty = swap_utils.torch.empty
    empty_calls = []

    def tracked_empty(*args, **kwargs):
        empty_calls.append((args, kwargs))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(swap_utils.torch, "empty", tracked_empty)

    manager = swap_utils.SwapBufferManager(num_elems=8, count=2, dtype=torch.float32, lazy=True)
    element_size = torch.tensor([], dtype=torch.float32).element_size()

    assert len(empty_calls) == 0
    assert manager.status()["total_bytes"] == 0
    assert manager.status()["capacity_bytes"] == 8 * 2 * element_size

    buffers = manager.allocate(num_elems=4, count=1, dtype=torch.float32)
    assert buffers is not None
    assert len(empty_calls) == 1
    assert manager.status()["total_bytes"] == 4 * element_size
    assert manager.status()["used_requested_bytes"] == 4 * element_size
    assert manager.status()["num_buffer_allocations"] == 1
    manager.free(buffers)

    reused_buffers = manager.allocate(num_elems=2, count=1, dtype=torch.float32)
    assert reused_buffers is not None
    assert len(empty_calls) == 1
    assert manager.status()["total_bytes"] == 4 * element_size
    assert manager.status()["used_requested_bytes"] == 2 * element_size
    manager.free(reused_buffers)


def test_swap_buffer_lease_releases_once(monkeypatch):
    _patch_swap_buffer_manager_deps(monkeypatch)

    manager = swap_utils.SwapBufferManager(num_elems=8, count=2, dtype=torch.float32)

    lease = manager.allocate_lease(num_elems=4, count=1, dtype=torch.float32, owner="unit lease")
    assert lease is not None
    assert len(lease) == 1
    assert manager.status()["free_buffer_count"] == 1
    assert manager.status()["used_buffer_count"] == 1

    lease.release()
    assert manager.status()["free_buffer_count"] == 2
    assert manager.status()["used_buffer_count"] == 0

    with pytest.raises(RuntimeError, match="released more than once"):
        lease.release()


def test_split_swap_buffer_counts_preserves_total_pipeline_budget():
    assert split_swap_buffer_counts(SimpleNamespace(buffer_count=4, pipeline=False)) == (4, 0)
    assert split_swap_buffer_counts(SimpleNamespace(buffer_count=12, pipeline=True)) == (8, 4)
    assert split_swap_buffer_counts(SimpleNamespace(buffer_count=16, pipeline=True)) == (12, 4)


def test_split_swap_buffer_counts_rejects_missing_state_pool():
    with pytest.raises(ValueError, match="Pipeline swap requires more than 4 total buffers"):
        split_swap_buffer_counts(SimpleNamespace(buffer_count=4, pipeline=True))


def test_optimizer_staging_swap_out_chunks_large_tensor(monkeypatch):
    monkeypatch.setattr(optimizer_swap_utils.dist, "get_rank", lambda: 1)

    swapper = OptimizerSwapper.__new__(OptimizerSwapper)
    swapper.numel_alignment = 1
    swapper.swap_element_size = torch.tensor([], dtype=torch.float32).element_size()
    swapper.staging_num_write_calls = 0
    swapper.staging_num_chunks_written = 0
    swapper.staging_num_elements_written = 0

    aio_handle = _FakeAIOHandle()
    src_tensor = torch.arange(10, dtype=torch.float32)
    pinned_buffers = [torch.empty(4, dtype=torch.float32), torch.empty(4, dtype=torch.float32)]

    swap_out_count = swapper._swap_out_unpinned_tensors(aio_handle=aio_handle,
                                                        unpinned_tensors=[src_tensor],
                                                        dest_paths=["tensor.swp"],
                                                        pinned_buffers=pinned_buffers)

    assert swap_out_count == 1
    assert [write[2] for write in aio_handle.writes] == [0, 16, 32]
    assert [write[0].numel() for write in aio_handle.writes] == [4, 4, 2]
    assert torch.equal(aio_handle.writes[0][0], src_tensor[:4])
    assert torch.equal(aio_handle.writes[1][0], src_tensor[4:8])
    assert torch.equal(aio_handle.writes[2][0], src_tensor[8:])
    assert aio_handle.wait_counts == [2, 1]
    assert swapper.staging_num_write_calls == 1
    assert swapper.staging_num_chunks_written == 3
    assert swapper.staging_num_elements_written == 10
