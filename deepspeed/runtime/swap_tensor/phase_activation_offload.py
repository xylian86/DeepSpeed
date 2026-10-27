# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Offload saved activations to a phase arena instead of recomputing them (SuperRL-Pin).

In the forward pass of an update micro-batch, every saved CUDA tensor of at
least ``min_bytes`` that is not a parameter is copied to a slice of a
PhasePinnedArena on a side stream. A tensor saved several times (for example
the input of the q, k and v projections) is copied once. In the backward pass,
unpacking a tensor also prefetches the next ones, in reverse pack order, until
``prefetch_bytes`` are on their way back, so the copies to HBM run ahead of the
backward kernels that need them.

Use it as ``with offload: loss = model(...)`` followed by ``loss.backward()``.
Leaving the ``with`` block ends the forward pass and starts the prefetch order.
Entering it again for the next micro-batch means the previous backward pass is
done, so its tensors are dropped and the arena is recycled: all copies to and
from the arena run on the one side stream, so the new copies into a slice queue
behind the copy that read it. Unpacking does not depend on autograd callbacks,
so autograd runs nested inside the forward pass (Liger's fused loss computes
its gradients there) and repeated unpacks of one tensor both work. The trainer
calls ``end_phase()`` when the policy update ends (before rollout), which
unpins the arena, and ``prepare_phase()`` when rollout ends, which starts
pinning one chunk of the size the last update used in a background thread, so
the pinning overlaps the work between rollout and the first forward pass.
"""

import os
import time
import weakref

import torch
from torch.autograd.graph import saved_tensors_hooks

from deepspeed.runtime.swap_tensor.phase_pinned_arena import PhasePinnedArena
from deepspeed.runtime.swap_tensor.utils import print_rank_0


def _is_param(tensor):
    return isinstance(tensor, torch.nn.Parameter) or hasattr(tensor, "ds_id")


class _Offloaded:
    __slots__ = ("host", "size", "stride", "dtype", "nbytes", "refs", "gpu", "ready", "source", "version")

    def __init__(self, host, size, stride, dtype, nbytes, source, version):
        self.host = host
        self.size = size
        self.stride = stride
        self.dtype = dtype
        self.nbytes = nbytes
        self.refs = 1  # packs not yet unpacked; the copy in HBM is dropped when it reaches 0
        self.gpu = None
        self.ready = None
        self.source = source  # weak reference to the packed tensor
        self.version = version

    def live_source(self):
        """The packed tensor itself, if it is still alive and unmodified."""
        tensor = self.source()
        return tensor if tensor is not None and tensor._version == self.version else None


class PhaseActivationOffload(saved_tensors_hooks):
    """``saved_tensors_hooks`` that keep saved activations in a native-pinned phase arena."""

    def __init__(self, min_bytes=1 << 20, prefetch_bytes=8 << 30, chunk_bytes=4 << 30, numa_node=None):
        self.min_bytes = min_bytes
        self.prefetch_bytes = prefetch_bytes
        self.arena = PhasePinnedArena("activations", chunk_bytes=chunk_bytes, registered=True, numa_node=numa_node)
        # SUPERRL_PROFILE_UPDATE=1: time the hooks on the CPU, and on the GPU how long the compute stream waits
        # for copies back to HBM and how far the copies to DRAM trail the end of each forward pass.
        self.profile = os.environ.get("SUPERRL_PROFILE_UPDATE") == "1"
        self._prof_waits = []  # (before, after) timing events around compute-stream waits
        self._prof_tails = []  # (forward end on the compute stream, last copy to DRAM issued by then)
        self._prof_cpu = {"pack": 0.0, "unpack": 0.0}
        self._prof_calls = {"pack": [], "unpack": [], "fetch alloc": [], "fetch copy": []}  # per-call CPU seconds
        self._prof_counts = {"offloaded": 0, "kept": 0, "deduplicated": 0}
        self._prof_mem = None  # allocator counters at the start of the phase
        self.stream = torch.cuda.Stream()
        self._entries = {}  # id -> _Offloaded, for the current micro-batch
        self._order = []  # offloaded ids in pack order
        self._by_key = {}  # de-duplication key -> id, during the forward pass
        self._cursor = None  # index in _order of the next tensor to prefetch; None during the forward pass
        self._inflight = 0  # bytes copied back to HBM and not yet unpacked
        self._next_id = 0
        self._last_phase_bytes = 0
        self._reset_stats()
        super().__init__(self._pack, self._unpack)

    def _reset_stats(self):
        self._micro_batches = 0
        self._micro_bytes = 0
        self._peak_micro_bytes = 0
        self._total_bytes = 0
        self._total_tensors = 0
        self._dedup_bytes = 0

    def _offloadable(self, tensor, span):
        if tensor.device.type != "cuda" or span * tensor.element_size() < self.min_bytes:
            return False
        if _is_param(tensor):
            return False
        return tensor._base is None or not _is_param(tensor._base)

    def _pack(self, tensor):
        if not self.profile:
            return self._pack_impl(tensor)
        if self._prof_mem is None:
            self._prof_mem = self._allocator_counters()
        start = time.perf_counter()
        try:
            return self._pack_impl(tensor)
        finally:
            elapsed = time.perf_counter() - start
            self._prof_cpu["pack"] += elapsed
            self._prof_calls["pack"].append(elapsed)

    def _pack_impl(self, tensor):
        # Copy the memory the tensor spans, gaps included, so it comes back with its own strides: kernels such
        # as memory-efficient attention require the aligned strides of a padded attention mask.
        size, stride = tuple(tensor.shape), tuple(tensor.stride())
        span = 1 + sum((n - 1) * st for n, st in zip(size, stride)) if tensor.numel() else 0
        if not self._offloadable(tensor, span):
            if self.profile:
                self._prof_counts["kept"] += 1
            return tensor
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
               tensor.dtype, tensor._version)
        known = self._by_key.get(key)
        if known is not None:
            entry = self._entries[known]
            if entry.live_source() is not None:
                entry.refs += 1
                self._dedup_bytes += entry.nbytes
                if self.profile:
                    self._prof_counts["deduplicated"] += 1
                return known

        if not self.arena.active and self._last_phase_bytes:
            self.arena.reserve(self._last_phase_bytes)
        nbytes = span * tensor.element_size()
        host = self.arena.allocate(span, tensor.dtype)
        compute = torch.cuda.current_stream()
        self.stream.wait_stream(compute)
        with torch.cuda.stream(self.stream):
            host.copy_(tensor.as_strided((span, ), (1, )), non_blocking=True)
        tensor.record_stream(self.stream)  # its HBM is not reused until the copy is done

        packed = self._next_id
        self._next_id += 1
        self._entries[packed] = _Offloaded(host, size, stride, tensor.dtype, nbytes, weakref.ref(tensor),
                                           tensor._version)
        self._order.append(packed)
        self._by_key[key] = packed
        self._micro_bytes += nbytes
        self._total_tensors += 1
        if self.profile:
            self._prof_counts["offloaded"] += 1
        return packed

    def _fetch(self, entry):
        with torch.cuda.stream(self.stream):
            if self.profile:
                t0 = time.perf_counter()
                flat = torch.empty(entry.host.numel(), dtype=entry.dtype, device="cuda")
                t1 = time.perf_counter()
                flat.copy_(entry.host, non_blocking=True)
                self._prof_calls["fetch alloc"].append(t1 - t0)
                self._prof_calls["fetch copy"].append(time.perf_counter() - t1)
            else:
                flat = torch.empty(entry.host.numel(), dtype=entry.dtype, device="cuda")
                flat.copy_(entry.host, non_blocking=True)
            entry.ready = self.stream.record_event()
        entry.gpu = flat.as_strided(entry.size, entry.stride)
        self._inflight += entry.nbytes

    def _prefetch(self):
        while self._cursor >= 0 and self._inflight < self.prefetch_bytes:
            entry = self._entries[self._order[self._cursor]]
            self._cursor -= 1
            if entry.gpu is None and entry.refs > 0 and entry.live_source() is None:
                self._fetch(entry)

    def _unpack(self, packed):
        if not self.profile:
            return self._unpack_impl(packed)
        start = time.perf_counter()
        try:
            return self._unpack_impl(packed)
        finally:
            elapsed = time.perf_counter() - start
            self._prof_cpu["unpack"] += elapsed
            self._prof_calls["unpack"].append(elapsed)

    @staticmethod
    def _allocator_counters():
        stats = torch.cuda.memory_stats()
        return {k: stats.get(k, 0) for k in ("num_device_alloc", "num_device_free", "num_alloc_retries")}

    def _timing_event(self, stream):
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    def _unpack_impl(self, packed):
        if isinstance(packed, torch.Tensor):
            return packed
        entry = self._entries[packed]
        if self._cursor is not None:
            self._prefetch()
        entry.refs -= 1
        source = entry.live_source()
        if source is not None and entry.gpu is None:
            return source  # still in HBM, for example during autograd run inside the forward pass
        if entry.gpu is None:
            self._fetch(entry)
        compute = torch.cuda.current_stream()
        if self.profile:
            before = self._timing_event(compute)
            compute.wait_event(entry.ready)
            self._prof_waits.append((before, self._timing_event(compute)))
        else:
            compute.wait_event(entry.ready)
        gpu = entry.gpu
        gpu.record_stream(compute)  # allocated on the side stream, used on the compute stream
        if entry.refs <= 0:
            entry.gpu = None  # autograd holds it while it needs it; a later unpack copies it back again
            self._inflight -= entry.nbytes
        return gpu

    def __enter__(self):
        if self._entries:
            self._end_micro_batch()
        return super().__enter__()

    def __exit__(self, *exc):
        super().__exit__(*exc)
        self._cursor = len(self._order) - 1  # the forward pass is over: prefetch from its last tensor
        self._by_key.clear()
        if self.profile and self._order:
            self._prof_tails.append((self._timing_event(torch.cuda.current_stream()), self._timing_event(self.stream)))

    def _end_micro_batch(self):
        """The previous micro-batch's backward pass is done: drop its tensors and recycle the arena."""
        for entry in self._entries.values():
            self.arena.free(entry.host)  # bookkeeping only
        self._entries.clear()
        self._order.clear()
        self._by_key.clear()
        self._cursor = None
        self._inflight = 0
        self._micro_batches += 1
        self._peak_micro_bytes = max(self._peak_micro_bytes, self._micro_bytes)
        self._total_bytes += self._micro_bytes
        self._micro_bytes = 0
        self.arena.recycle()

    def prepare_phase(self):
        """Rollout is over: start pinning the next update's arena in the background."""
        if self._last_phase_bytes and not self.arena.active:
            self.arena.reserve_async(self._last_phase_bytes)

    def end_phase(self):
        """End the policy update, after its last backward pass: unpin the arena. Returns the bytes it had pinned."""
        if self._entries:
            self._end_micro_batch()
        if not self.arena.active:
            return 0
        self.arena.release()  # synchronizes the device, so the profiling events below are complete
        self._last_phase_bytes = self.arena.last_phase_bytes
        n = max(self._micro_batches, 1)
        if self.profile:
            wait = sum(a.elapsed_time(b) for a, b in self._prof_waits) / 1e3
            tail = sum(max(f.elapsed_time(d), 0.0) for f, d in self._prof_tails) / 1e3
            print_rank_0(
                f"[superrl_profile] activation offload, {self._micro_batches} micro-batches: hooks on the CPU "
                f"pack {self._prof_cpu['pack']:.2f} s, unpack {self._prof_cpu['unpack']:.2f} s; compute stream waited "
                f"{wait:.2f} s for copies back to HBM ({len(self._prof_waits)} waits); copies to DRAM ended "
                f"{tail:.2f} s after the forward passes (sum)",
                force=True)
            for name, calls in self._prof_calls.items():
                if not calls:
                    continue
                calls.sort()
                q = lambda f: calls[min(int(f * len(calls)), len(calls) - 1)] * 1e6
                slow = [c for c in calls if c > 1e-3]
                print_rank_0(
                    f"[superrl_profile]   {name}: {len(calls)} calls, sum {sum(calls):.2f} s, p50 {q(0.5):.0f} us, "
                    f"p90 {q(0.9):.0f} us, p99 {q(0.99):.0f} us, max {calls[-1] * 1e6:.0f} us; "
                    f"{len(slow)} calls over 1 ms take {sum(slow):.2f} s",
                    force=True)
                calls.clear()
            mem = self._allocator_counters()
            delta = {k: mem[k] - (self._prof_mem or mem)[k] for k in mem}
            import threading
            print_rank_0(
                f"[superrl_profile]   saved tensors: {self._prof_counts}; allocator during the phase: {delta}; "
                f"Python threads: {sorted(t.name for t in threading.enumerate())}",
                force=True)
            self._prof_counts = dict.fromkeys(self._prof_counts, 0)
            self._prof_mem = None
            self._prof_waits.clear()
            self._prof_tails.clear()
            self._prof_cpu = {"pack": 0.0, "unpack": 0.0}
        print_rank_0(
            f"[superrl_pin] activation offload: {self._micro_batches} micro-batches, "
            f"{self._total_bytes / n / 2**30:.2f} GiB offloaded per micro-batch "
            f"(peak {self._peak_micro_bytes / 2**30:.2f} GiB, {self._total_tensors / n:.0f} tensors), "
            f"{self._dedup_bytes / n / 2**30:.2f} GiB per micro-batch saved by de-duplication",
            force=True)
        self._reset_stats()
        return self._last_phase_bytes
