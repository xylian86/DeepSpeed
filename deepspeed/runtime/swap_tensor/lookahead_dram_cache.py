# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Execution-ordered DRAM cache for NVMe-resident ZeRO parameter partitions.
"""

from dataclasses import dataclass
import math
import os
from typing import Callable, Dict, Iterable, Optional

import torch

from deepspeed.runtime.swap_tensor.utils import swap_in_tensors


HOST_MEMORY_CAP_FRACTION = 0.9


def _read_int(path):
    try:
        with open(path, "r") as handle:
            value = handle.read().strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _system_memory_bytes():
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return page_size * pages


def _cgroup_memory_limit_bytes():
    candidates = [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    system_memory = _system_memory_bytes()
    for path in candidates:
        limit = _read_int(path)
        if limit is not None and limit > 0 and limit < (1 << 60):
            return min(limit, system_memory)
    return system_memory


def _cgroup_memory_current_bytes():
    candidates = [
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ]
    for path in candidates:
        current = _read_int(path)
        if current is not None and current >= 0:
            return current
    return None


def _local_world_size():
    for name in ("LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            value = int(value)
        except ValueError:
            continue
        if value > 0:
            return value
    return 1


def effective_host_memory_limit_bytes():
    return min(_system_memory_bytes(), _cgroup_memory_limit_bytes())


def process_rss_bytes():
    with open("/proc/self/statm", "r") as handle:
        statm = handle.read().split()
    resident_pages = int(statm[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _status_name(param):
    status = getattr(getattr(param, "ds_tensor", None), "status", None)
    return getattr(status, "name", str(status))


def _is_not_available(param):
    return _status_name(param) == "NOT_AVAILABLE"


@dataclass
class _CacheEntry:
    param: object
    buffer: torch.Tensor
    aligned_numel: int
    numel: int
    path: str
    next_use_index: float
    status: str
    valid: bool = True

    @property
    def param_id(self):
        return self.param.ds_id

    @property
    def nbytes(self):
        return self.buffer.numel() * self.buffer.element_size()

    def compute_buffer(self):
        return self.buffer.narrow(0, 0, self.numel)


class LookaheadDRAMCache:
    """Greedy look-ahead cache bounded by 90% of host memory.

    The public DeepSpeed config is intentionally one boolean. Internally, the
    cache grows while the process RSS remains below 90% of the effective host
    memory limit and evicts entries whose future use is farthest away.
    """

    def __init__(self,
                 enabled,
                 dtype,
                 aio_handle,
                 pin_memory_fn,
                 aligned_numel_fn: Callable[[int], int],
                 host_memory_limit_bytes_fn=effective_host_memory_limit_bytes,
                 host_memory_current_bytes_fn=_cgroup_memory_current_bytes,
                 process_rss_bytes_fn=process_rss_bytes,
                 local_world_size_fn=_local_world_size,
                 max_prefetches_per_call=8,
                 max_inflight_prefetches=16):
        self.enabled = enabled
        self.dtype = dtype
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        self.aio_handle = aio_handle
        self.pin_memory_fn = pin_memory_fn
        self.aligned_numel_fn = aligned_numel_fn
        self.host_memory_limit_bytes_fn = host_memory_limit_bytes_fn
        self.host_memory_current_bytes_fn = host_memory_current_bytes_fn
        self.process_rss_bytes_fn = process_rss_bytes_fn
        self.local_world_size_fn = local_world_size_fn
        self.max_prefetches_per_call = max_prefetches_per_call
        self.max_inflight_prefetches = max_inflight_prefetches

        self.entries: Dict[int, _CacheEntry] = {}
        self.pending_reads = 0
        self.trace_len = 0

        self.hits = 0
        self.misses = 0
        self.prefetches = 0
        self.evictions = 0
        self.invalidations = 0
        self.bytes_read = 0
        self.inflight_waits = 0
        self.capacity_denials = 0

    @property
    def used_bytes(self):
        return sum(entry.nbytes for entry in self.entries.values())

    @property
    def global_memory_cap_bytes(self):
        return int(self.host_memory_limit_bytes_fn() * HOST_MEMORY_CAP_FRACTION)

    @property
    def local_world_size(self):
        return max(1, int(self.local_world_size_fn()))

    @property
    def memory_cap_bytes(self):
        return self.global_memory_cap_bytes // self.local_world_size

    def _entry_is_attached(self, entry):
        if _is_not_available(entry.param):
            return False
        tensor = getattr(entry.param, "ds_tensor", None)
        if tensor is None or tensor.numel() == 0:
            return False
        try:
            return tensor.data_ptr() == entry.compute_buffer().data_ptr()
        except RuntimeError:
            return False

    def _safe_to_evict(self, entry):
        return entry.status != "inflight" and not self._entry_is_attached(entry)

    def _evict(self, param_id):
        entry = self.entries.pop(param_id)
        del entry
        self.evictions += 1

    def _evict_invalid_entries(self):
        for param_id, entry in list(self.entries.items()):
            if not entry.valid and self._safe_to_evict(entry):
                self._evict(param_id)

    def _has_process_room_for(self, nbytes):
        if self.used_bytes + nbytes > self.memory_cap_bytes:
            return False

        global_memory_cap = self.global_memory_cap_bytes
        current = self.host_memory_current_bytes_fn()
        if current is not None and current + nbytes > global_memory_cap:
            return False

        return self.process_rss_bytes_fn() + nbytes <= global_memory_cap

    def _ensure_capacity_for(self, nbytes, candidate_next_use):
        self._evict_invalid_entries()
        while not self._has_process_room_for(nbytes):
            evictable = [entry for entry in self.entries.values() if self._safe_to_evict(entry)]
            if not evictable:
                self.capacity_denials += 1
                return False

            victim = max(evictable, key=lambda entry: entry.next_use_index)
            if victim.next_use_index <= candidate_next_use:
                self.capacity_denials += 1
                return False
            self._evict(victim.param_id)
        return True

    def _next_use_map(self, ordered_params):
        next_use = {}
        for index, param in enumerate(ordered_params):
            next_use.setdefault(param.ds_id, index)
        return next_use

    def prefetch(self, ordered_params: Iterable[object], path_fn: Callable[[object], str]):
        if not self.enabled:
            return

        ordered_params = list(ordered_params)
        self.trace_len = max(self.trace_len, len(ordered_params))
        next_use = self._next_use_map(ordered_params)
        for param_id, entry in self.entries.items():
            entry.next_use_index = next_use.get(param_id, math.inf)

        submitted = 0
        for param in ordered_params:
            if submitted >= self.max_prefetches_per_call or self.pending_reads >= self.max_inflight_prefetches:
                break
            if not _is_not_available(param):
                continue
            if param.ds_id in self.entries:
                continue

            numel = param.ds_tensor.ds_numel
            aligned_numel = self.aligned_numel_fn(numel)
            nbytes = aligned_numel * self.element_size
            candidate_next_use = next_use[param.ds_id]
            if not self._ensure_capacity_for(nbytes, candidate_next_use):
                break

            path = path_fn(param)
            buffer = torch.empty(aligned_numel, device="cpu", dtype=self.dtype)
            buffer = self.pin_memory_fn(buffer, align_bytes=0)
            entry = _CacheEntry(param=param,
                                buffer=buffer,
                                aligned_numel=aligned_numel,
                                numel=numel,
                                path=path,
                                next_use_index=candidate_next_use,
                                status="inflight")
            self.entries[param.ds_id] = entry
            self.pending_reads += swap_in_tensors(self.aio_handle, [entry.buffer], [entry.path])
            self.prefetches += 1
            self.bytes_read += entry.nbytes
            submitted += 1

    def synchronize_reads(self):
        if self.pending_reads == 0:
            return
        assert self.aio_handle.wait() == self.pending_reads
        self.pending_reads = 0
        for entry in self.entries.values():
            if entry.status == "inflight":
                entry.status = "ready"
        self._evict_invalid_entries()

    def acquire(self, param) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None

        entry = self.entries.get(param.ds_id)
        if entry is None or not entry.valid:
            self.misses += 1
            return None
        if entry.status == "inflight":
            self.inflight_waits += 1
            self.synchronize_reads()
        if entry.status != "ready" or not entry.valid:
            self.misses += 1
            return None

        self.hits += 1
        return entry.compute_buffer()

    def invalidate(self, params):
        for param in params:
            entry = self.entries.get(param.ds_id)
            if entry is None:
                continue
            entry.valid = False
            self.invalidations += 1
        self._evict_invalid_entries()

    def on_param_detached(self, param):
        entry = self.entries.get(param.ds_id)
        if entry is not None and not entry.valid and self._safe_to_evict(entry):
            self._evict(param.ds_id)

    def stats(self):
        requests = self.hits + self.misses
        hit_ratio = float(self.hits) / requests if requests else 0.0
        return {
            "superrl_cache/hit_ratio": hit_ratio,
            "superrl_cache/hits": self.hits,
            "superrl_cache/misses": self.misses,
            "superrl_cache/used_bytes": self.used_bytes,
            "superrl_cache/trace_len": self.trace_len,
            "superrl_cache/prefetches": self.prefetches,
            "superrl_cache/evictions": self.evictions,
            "superrl_cache/invalidations": self.invalidations,
            "superrl_cache/bytes_read": self.bytes_read,
            "superrl_cache/inflight_waits": self.inflight_waits,
            "superrl_cache/capacity_denials": self.capacity_denials,
            "superrl_cache/memory_cap_bytes": self.memory_cap_bytes,
            "superrl_cache/global_memory_cap_bytes": self.global_memory_cap_bytes,
            "superrl_cache/local_world_size": self.local_world_size,
        }
