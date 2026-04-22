# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LookaheadDRAMCache (Belady eviction + position tracking)."""
import torch

from deepspeed.runtime.superrl.cache.lookahead_cache import LookaheadDRAMCache


def _t(numel, dtype=torch.float32):
    return torch.zeros(numel, dtype=dtype)


def test_lookup_miss_then_insert_then_hit():
    cache = LookaheadDRAMCache(dram_budget_bytes=1 << 20)
    assert cache.lookup(1) is None
    cache.insert(1, _t(8))
    assert cache.lookup(1) is not None
    s = cache.stats()
    assert s["superrl_cache/hits"] == 1
    assert s["superrl_cache/misses"] == 1


def test_belady_evicts_furthest_next_use():
    """Eviction should drop the param whose next use is furthest in the
    future (or never used again) according to the trace + cursor."""
    cache = LookaheadDRAMCache(dram_budget_bytes=2 * 8 * 4)  # room for two 8-elem fp32 tensors
    # Trace: 1 used at 0 and 3; 2 used at 1 (never reused); 3 used at 2.
    cache.update_trace([1, 2, 3, 1])

    cache.insert(1, _t(8))
    cache.insert(2, _t(8))
    # At cursor=0: next_use(1)=0, next_use(2)=1. Belady evicts 2 (farther).
    cache.insert(3, _t(8))

    assert cache.lookup(2) is None  # evicted
    assert cache.lookup(1) is not None
    assert cache.lookup(3) is not None


def test_belady_evicts_never_reused_first():
    """Items with no future use should be evicted before items that will
    be touched again."""
    cache = LookaheadDRAMCache(dram_budget_bytes=2 * 8 * 4)
    # 99 has no next use after position 1; 1 will be reused at position 3.
    cache.update_trace([1, 99, 2, 1, 2])

    cache.insert(1, _t(8))
    cache.insert(99, _t(8))
    # Force eviction; 99 (next_use = end-of-trace) loses to 1 (next_use=3).
    cache.insert(2, _t(8))
    assert cache.lookup(99) is None
    assert cache.lookup(1) is not None


def test_cursor_advances_past_lookups():
    cache = LookaheadDRAMCache(dram_budget_bytes=1 << 20)
    cache.update_trace([10, 20, 30, 10, 20, 30])
    assert cache.stats()["superrl_cache/cursor"] == 0
    cache.lookup(10)
    assert cache.stats()["superrl_cache/cursor"] == 1
    cache.lookup(20)
    assert cache.stats()["superrl_cache/cursor"] == 2


def test_repeated_pass_yields_hits():
    """Pass over a trace twice; second pass should hit on cached entries."""
    cache = LookaheadDRAMCache(dram_budget_bytes=1 << 20)
    trace = [1, 2, 3, 1, 2, 3]
    cache.update_trace(trace)
    for pid in trace[:3]:
        cache.lookup(pid)
        cache.insert(pid, _t(8))
    # Second pass:
    cache.reset_position()
    hits_before = cache.stats()["superrl_cache/hits"]
    for pid in trace[:3]:
        cache.lookup(pid)
    hits_after = cache.stats()["superrl_cache/hits"]
    assert hits_after - hits_before == 3


def test_oversized_tensor_is_skipped():
    cache = LookaheadDRAMCache(dram_budget_bytes=16)
    big = _t(1024)  # 4KB - far over budget
    cache.insert(1, big)
    assert cache.lookup(1) is None  # skipped


def test_evict_decrements_used_bytes():
    cache = LookaheadDRAMCache(dram_budget_bytes=1 << 20)
    cache.insert(1, _t(8))
    used_before = cache.stats()["superrl_cache/used_bytes"]
    cache.evict(1)
    used_after = cache.stats()["superrl_cache/used_bytes"]
    assert used_after < used_before
    assert cache.lookup(1) is None
