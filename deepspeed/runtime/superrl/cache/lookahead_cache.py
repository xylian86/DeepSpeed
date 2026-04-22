# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""DRAM look-ahead cache for NVMe-offloaded parameters.

Paper sec. IV.B: SuperRL-Cache turns Grace DRAM into a look-ahead cache
that follows the model's execution order. Before each access we have the
chance to prefetch upcoming parameters; on eviction we use Belady
(furthest-next-use) since the trace makes the future known.

Hit ratio target from the paper: ~36-40% for both dense and MoE models.
"""

import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch


class LookaheadDRAMCache:
    """DRAM-resident parameter cache with look-ahead prefetch.

    Parameters
    ----------
    dram_budget_bytes:
        Maximum number of bytes to keep in DRAM at any time.
    nvme_engine:
        ``CoalescedNVMeEngine`` used to async-prefetch from NVMe into DRAM.
        May be ``None`` for a memory-only mode (tests).
    trace:
        Ordered list of ``param_id``s representing the execution-ordered
        access sequence produced by ``TraceRecorder``.
    window_size:
        How many trace positions ahead to consider when prefetching
        (``0`` = "fill until DRAM budget exhausted").
    """

    def __init__(
        self,
        dram_budget_bytes: int,
        nvme_engine=None,
        trace: Optional[List[int]] = None,
        window_size: int = 0,
    ) -> None:
        self.dram_budget_bytes = int(dram_budget_bytes)
        self.nvme_engine = nvme_engine
        self.window_size = int(window_size)
        self._trace: List[int] = list(trace) if trace else []
        # Inverted index of next occurrences for fast Belady queries.
        self._positions: Dict[int, List[int]] = self._build_positions(self._trace)
        self._cursor: int = 0

        # Storage: param_id -> (cpu_pinned_tensor, byte_size)
        self._cache: "OrderedDict[int, Tuple[torch.Tensor, int]]" = OrderedDict()
        self._used_bytes: int = 0
        self._lock = threading.RLock()

        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    # Trace handling
    # ------------------------------------------------------------------

    @staticmethod
    def _build_positions(trace: List[int]) -> Dict[int, List[int]]:
        positions: Dict[int, List[int]] = {}
        for idx, pid in enumerate(trace):
            positions.setdefault(pid, []).append(idx)
        return positions

    def update_trace(self, trace: List[int]) -> None:
        """Replace the access trace (called once warm-up completes)."""
        with self._lock:
            self._trace = list(trace)
            self._positions = self._build_positions(self._trace)
            self._cursor = 0

    def reset_position(self) -> None:
        """Restart at the beginning of the trace (one call per training step)."""
        with self._lock:
            self._cursor = 0

    # ------------------------------------------------------------------
    # Lookup / insert / evict
    # ------------------------------------------------------------------

    def lookup(self, param_id: int) -> Optional[torch.Tensor]:
        """Return the cached pinned tensor for ``param_id`` or ``None``.

        Advances the trace cursor past the matched position so that future
        Belady queries reflect the new "future".
        """
        with self._lock:
            entry = self._cache.get(param_id)
            self._advance_cursor_past(param_id)
            if entry is not None:
                self.hits += 1
                tensor, _ = entry
                return tensor
            self.misses += 1
            return None

    def insert(self, param_id: int, tensor: torch.Tensor) -> None:
        """Add ``tensor`` to the cache, evicting via Belady to make room."""
        nbytes = tensor.numel() * tensor.element_size()
        with self._lock:
            if param_id in self._cache:
                return
            if nbytes > self.dram_budget_bytes:
                return  # too large to ever cache
            self._evict_to_fit(nbytes)
            cpu_tensor = self._to_pinned_cpu(tensor)
            self._cache[param_id] = (cpu_tensor, nbytes)
            self._used_bytes += nbytes

    def evict(self, param_id: int) -> None:
        with self._lock:
            entry = self._cache.pop(param_id, None)
            if entry is not None:
                self._used_bytes -= entry[1]

    # ------------------------------------------------------------------
    # Look-ahead prefetch
    # ------------------------------------------------------------------

    def prefetch_window(
        self,
        param_paths: Dict[int, str],
        param_shapes: Dict[int, Tuple[int, ...]],
        param_dtype: Dict[int, torch.dtype],
    ) -> int:
        """Async-prefetch upcoming params into DRAM up to the budget/window.

        Walks the trace forward from the current cursor, skipping already
        cached entries and entries whose metadata we lack, until the DRAM
        budget is exhausted or the window cap is reached. Returns the number
        of prefetch IORequests submitted to the engine.
        """
        if self.nvme_engine is None or not self._trace:
            return 0

        from ..io.nvme_engine import IORequest

        with self._lock:
            requests: List[IORequest] = []
            seen: set = set(self._cache.keys())
            budget_remaining = self.dram_budget_bytes - self._used_bytes
            window_end = (
                len(self._trace) if self.window_size <= 0
                else min(len(self._trace), self._cursor + self.window_size)
            )

            for idx in range(self._cursor, window_end):
                pid = self._trace[idx]
                if pid in seen:
                    continue
                seen.add(pid)
                if pid not in param_paths or pid not in param_shapes:
                    continue
                dtype = param_dtype.get(pid, torch.float32)
                numel = 1
                for s in param_shapes[pid]:
                    numel *= s
                nbytes = numel * torch.tensor([], dtype=dtype).element_size()
                if nbytes <= 0 or nbytes > self.dram_budget_bytes:
                    continue
                if nbytes > budget_remaining:
                    # Try Belady eviction to make room.
                    self._evict_to_fit(nbytes)
                    budget_remaining = self.dram_budget_bytes - self._used_bytes
                    if nbytes > budget_remaining:
                        continue

                buf = self.nvme_engine.allocate_host_buffer(numel, dtype)
                requests.append(IORequest(buffer=buf, path=param_paths[pid], group="cache_prefetch"))
                self._cache[pid] = (buf, nbytes)
                self._used_bytes += nbytes
                budget_remaining -= nbytes

        if requests:
            self.nvme_engine.submit_reads(requests)
        return len(requests)

    # ------------------------------------------------------------------
    # Belady internals
    # ------------------------------------------------------------------

    def _advance_cursor_past(self, param_id: int) -> None:
        """Advance the cursor to the access *after* the next occurrence of pid.

        We use the precomputed positions to find the next occurrence at or
        after the current cursor; if found, the cursor jumps past it. This
        keeps Belady's "next-use" calculation correct without scanning the
        whole trace each time.
        """
        positions = self._positions.get(param_id)
        if not positions:
            return
        # Binary search for the first index >= self._cursor.
        lo, hi = 0, len(positions)
        while lo < hi:
            mid = (lo + hi) // 2
            if positions[mid] < self._cursor:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(positions):
            self._cursor = positions[lo] + 1

    def _next_use(self, param_id: int) -> int:
        """Return the trace position of the next use of ``param_id`` at or
        after the current cursor; ``len(trace)`` if it is never used again."""
        positions = self._positions.get(param_id)
        if not positions:
            return len(self._trace)
        lo, hi = 0, len(positions)
        while lo < hi:
            mid = (lo + hi) // 2
            if positions[mid] < self._cursor:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(positions):
            return positions[lo]
        return len(self._trace)

    def _evict_to_fit(self, needed_bytes: int) -> None:
        """Belady eviction loop: while we don't fit, drop the param whose
        next use is farthest in the future."""
        while self._used_bytes + needed_bytes > self.dram_budget_bytes and self._cache:
            victim = self._belady_victim()
            entry = self._cache.pop(victim, None)
            if entry is None:
                break
            self._used_bytes -= entry[1]

    def _belady_victim(self) -> int:
        worst_pid: Optional[int] = None
        worst_pos: int = -1
        for pid in self._cache:
            pos = self._next_use(pid)
            if pos > worst_pos:
                worst_pos = pos
                worst_pid = pid
        if worst_pid is None:
            # Fallback to oldest-inserted; should not happen if cache non-empty.
            worst_pid = next(iter(self._cache))
        return worst_pid

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pinned_cpu(tensor: torch.Tensor) -> torch.Tensor:
        """Copy ``tensor`` into a host-pinned tensor (the DRAM tier).

        D2H is issued non_blocking and we synchronize the current stream
        before returning, so the returned pinned tensor is guaranteed
        valid even if the caller recycles the source GPU buffer
        immediately afterwards. This matches the safety contract that
        ``partitioned_param_swapper.synchronize_reads`` relies on when
        it inserts post-NVMe-read GPU buffers into the cache.
        """
        if tensor.is_cuda:
            try:
                pinned = torch.empty(
                    tensor.numel(), dtype=tensor.dtype, pin_memory=True
                )
            except (RuntimeError, NotImplementedError):
                pinned = torch.empty(tensor.numel(), dtype=tensor.dtype)
            pinned.copy_(tensor.detach().view(-1), non_blocking=True)
            torch.cuda.current_stream(tensor.device).synchronize()
            return pinned
        cpu = tensor.detach().clone()
        try:
            return cpu.pin_memory()
        except (RuntimeError, NotImplementedError):
            return cpu

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            hit_ratio = self.hits / total if total else 0.0
            return {
                "superrl_cache/hits": self.hits,
                "superrl_cache/misses": self.misses,
                "superrl_cache/hit_ratio": hit_ratio,
                "superrl_cache/used_bytes": self._used_bytes,
                "superrl_cache/budget_bytes": self.dram_budget_bytes,
                "superrl_cache/cursor": self._cursor,
                "superrl_cache/trace_len": len(self._trace),
            }
