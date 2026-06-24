# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.runtime.swap_tensor import utils as swap_utils


class _FakeAccelerator:

    def pin_memory(self, tensor, align_bytes=0):
        return tensor


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


def test_swap_buffer_manager_reports_allocation_state(monkeypatch):
    _patch_swap_buffer_manager_deps(monkeypatch)

    manager = swap_utils.SwapBufferManager(num_elems=8, count=2, dtype=torch.float32)

    buffers = manager.allocate(num_elems=4, count=1, dtype=torch.float32)
    assert buffers is not None
    assert manager.status()["free_buffer_count"] == 1
    assert manager.status()["used_buffer_count"] == 1
    assert manager.status()["max_allocated_buffers"] == 1

    assert manager.allocate(num_elems=4, count=2, dtype=torch.float32) is None
    status = manager.status()
    assert status["num_failed_allocations"] == 1
    assert status["max_requested_count"] == 2

    message = manager.allocation_failure_message(requested_num_elems=4,
                                                 requested_count=2,
                                                 owner="unit test")
    assert "unit test" in message
    assert "free_buffer_count=1" in message
    assert "used_buffer_count=1" in message
    assert "buffer_count=2" in message

    manager.free(buffers)
    assert manager.status()["free_buffer_count"] == 2
