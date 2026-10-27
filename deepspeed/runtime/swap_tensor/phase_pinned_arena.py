# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Pinned host memory that lives for one training phase (SuperRL-Pin).

Unpinning page-locked memory waits for queued GPU work, so freeing pinned
buffers one at a time while the GPU computes stalls the step. A phase arena
pins a few large chunks with the native backend, hands out slices of them,
and unpins the chunks together at the phase boundary, when the GPU has
drained. Freeing a slice is bookkeeping only.

With ``registered=True`` a chunk is an anonymous mapping whose pages are
faulted in by PyTorch's intra-op threads and then registered with CUDA
(cudaHostRegister), without mlock. That pins about 6x faster than the native
backend, whose single-threaded mlock holds the GIL, and ``reserve_async()`` can
pin it in a background thread; only the registration itself blocks the main
thread's CUDA calls (about 4 ms per GiB). Chunks are unmapped in a background
thread. Such chunks are not in DeepNVMe's pinned-tensor table, so use it only
for buffers that never go to NVMe.

A registered arena can also bind its chunks to one NUMA node, by default the
node of the CPUs this process runs on (``numa_node="local"``). With
zone_reclaim_mode 0 the kernel otherwise places pages on another node instead
of reclaiming page cache on the local one; with the binding it reclaims. The
binding is skipped, with a warning, if the node's free memory plus file pages
cannot hold the chunk.
"""

import collections
import ctypes
import glob
import mmap
import os
import platform
import threading
import time

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.swap_tensor.utils import print_rank_0
from deepspeed.utils.pin_memory import get_active_native_pinned_memory

ARENA_ALIGNMENT_BYTES = 4096


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


_SYSCALLS = {"aarch64": {"mbind": 235, "get_mempolicy": 236}, "x86_64": {"mbind": 237, "get_mempolicy": 239}}
_MPOL_BIND = 2
_MPOL_F_NODE_ADDR = 3  # MPOL_F_NODE | MPOL_F_ADDR
_NODE_MASK_BITS = 1024
_BIND_MARGIN_BYTES = 8 << 30


def local_numa_node():
    """The NUMA node of the first CPU this process may run on, or None."""
    cpu = min(os.sched_getaffinity(0))
    for path in glob.glob("/sys/devices/system/node/node[0-9]*/cpulist"):
        for part in open(path).read().strip().split(","):
            if part:
                lo, _, hi = part.partition("-")
                if int(lo) <= cpu <= int(hi or lo):
                    return int(path.split("/")[-2][4:])
    return None


def _node_reclaimable_bytes(node):
    """Free memory plus file pages on ``node``: what a bound allocation can get by reclaiming page cache."""
    fields = {}
    for line in open(f"/sys/devices/system/node/node{node}/meminfo"):
        parts = line.split()
        fields[parts[2].rstrip(":")] = int(parts[3]) * 1024
    return fields["MemFree"] + fields.get("Active(file)", 0) + fields.get("Inactive(file)", 0)


def _syscall(name, *args):
    number = _SYSCALLS.get(platform.machine(), {}).get(name)
    if number is None:
        raise OSError(f"no {name} syscall number for {platform.machine()}")
    if _libc.syscall(number, *args) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"{name}: {os.strerror(errno)}")


def _bind_range(addr, nbytes, node):
    mask = (ctypes.c_ulong * (_NODE_MASK_BITS // 64))()
    mask[node // 64] = 1 << (node % 64)
    _syscall("mbind", ctypes.c_void_p(addr), ctypes.c_ulong(nbytes), ctypes.c_int(_MPOL_BIND), mask,
             ctypes.c_ulong(_NODE_MASK_BITS + 1), ctypes.c_uint(0))


def _sample_nodes(addr, nbytes, samples=64):
    """Count the NUMA nodes of ``samples`` evenly spaced (faulted-in) pages of a range."""
    counts = collections.Counter()
    node = ctypes.c_int()
    for i in range(samples):
        page = addr + (nbytes - 1) * i // max(samples - 1, 1)
        try:
            _syscall("get_mempolicy", ctypes.byref(node), None, ctypes.c_ulong(0), ctypes.c_void_p(page),
                     ctypes.c_ulong(_MPOL_F_NODE_ADDR))
            counts[node.value] += 1
        except OSError:
            counts["?"] += 1
    return counts


def _map_registered(nbytes, node):
    addr = _libc.mmap(None, nbytes, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    if addr is None or addr == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), f"mmap of {nbytes} bytes failed")
    if node is not None:
        _bind_range(addr, nbytes, node)
    tensor = torch.frombuffer((ctypes.c_uint8 * nbytes).from_address(addr), dtype=torch.uint8)
    tensor.fill_(0)  # fault the pages in on the intra-op threads; ctypes calls and torch ops drop the GIL
    get_accelerator().register_host_memory(addr, nbytes)
    return tensor


def _unmap_registered(tensor):
    addr, nbytes = tensor.data_ptr(), tensor.numel()
    get_accelerator().unregister_host_memory(addr)
    threading.Thread(target=_libc.munmap, args=(addr, nbytes)).start()


class _Chunk:

    def __init__(self, tensor):
        self.tensor = tensor
        self.nbytes = tensor.numel()
        self.next_offset = 0
        self.live = 0
        self.closing = False


class PhasePinnedArena:
    """Bump allocator over native-pinned chunks, released once per phase.

    Slices are not reused within a phase unless the caller calls
    ``recycle()``, so a slice handed out is not overwritten while GPU copies
    from an earlier owner are still queued.
    ``release()`` synchronizes the device and unpins every chunk without live
    slices; a chunk that still has live slices is unpinned when the last one is
    freed.
    """

    def __init__(self, name, chunk_bytes=1 << 30, alignment=ARENA_ALIGNMENT_BYTES, registered=False, numa_node=None):
        self.name = name
        self.chunk_bytes = chunk_bytes
        self.alignment = alignment
        self.registered = registered
        self.numa_node = local_numa_node() if numa_node == "local" else numa_node
        self.phase_placement = collections.Counter()  # sampled pages per NUMA node in this phase's chunks
        self.phase_unbound_chunks = 0
        self._native = get_active_native_pinned_memory()
        if self._native is None and not registered:
            raise RuntimeError("PhasePinnedArena requires DS_PIN_MEMORY_BACKEND=native")
        self._reserving = None  # background thread started by reserve_async()
        self._reserve_error = None
        self.last_wait_seconds = 0.0
        self._chunks = []
        self._closing = []
        self._slices = {}

        self.phases = 0
        self.reserved_bytes = 0
        self.peak_reserved_bytes = 0
        self.phase_pin_seconds = 0.0
        self.last_pin_seconds = 0.0
        self.last_unpin_seconds = 0.0
        self.last_phase_bytes = 0

    @property
    def active(self):
        """True while the current phase holds pinned chunks."""
        self._wait_reserved()
        return bool(self._chunks)

    def owns(self, buffer):
        return buffer.data_ptr() in self._slices

    @property
    def used_bytes(self):
        return sum(nbytes for _, nbytes in self._slices.values())

    def reserve(self, nbytes):
        """Pin one chunk of ``nbytes`` (rounded to the alignment) for this phase."""
        return self._pin_chunk(_round_up(nbytes, self.alignment))

    def reserve_async(self, nbytes):
        """Start reserving ``nbytes`` in a background thread; the next allocate() or release() waits for it."""
        if self._reserving is not None:
            return

        def run():
            try:
                self.reserve(nbytes)
            except BaseException as e:  # re-raised by _wait_reserved()
                self._reserve_error = e

        # Not a daemon: interrupting cudaHostRegister at interpreter exit aborts the process.
        self._reserving = threading.Thread(target=run, name=f"{self.name}-arena-pin")
        self._reserving.start()

    def _wait_reserved(self):
        if self._reserving is None:
            return
        start = time.perf_counter()
        self._reserving.join()
        self.last_wait_seconds += time.perf_counter() - start
        self._reserving = None
        if self._reserve_error is not None:
            error, self._reserve_error = self._reserve_error, None
            raise error

    def _pin_chunk(self, nbytes):
        start = time.perf_counter()
        if self.registered:
            node = self.numa_node
            if node is not None and _node_reclaimable_bytes(node) < nbytes + _BIND_MARGIN_BYTES:
                print_rank_0(f"[superrl_pin] {self.name} arena: node {node} cannot hold a "
                             f"{nbytes / 2**30:.2f} GiB chunk; not binding it", force=True)
                node = None
                self.phase_unbound_chunks += 1
            tensor = _map_registered(nbytes, node)
            self.phase_placement.update(_sample_nodes(tensor.data_ptr(), nbytes))
        else:
            tensor = self._native.pin_empty(torch.empty(0, dtype=torch.uint8), (nbytes, ))
        self.phase_pin_seconds += time.perf_counter() - start
        chunk = _Chunk(tensor)
        self._chunks.append(chunk)
        self.reserved_bytes += nbytes
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)
        return chunk

    def allocate(self, numel, dtype):
        self._wait_reserved()
        nbytes = numel * torch.tensor([], dtype=dtype).element_size()
        aligned = _round_up(max(nbytes, 1), self.alignment)
        chunk = next((c for c in self._chunks if c.nbytes - c.next_offset >= aligned), None)
        if chunk is None:
            chunk = self._pin_chunk(max(self.chunk_bytes, aligned))
        offset = chunk.next_offset
        chunk.next_offset += aligned
        chunk.live += 1
        buffer = chunk.tensor.narrow(0, offset, nbytes).view(dtype)
        self._slices[buffer.data_ptr()] = (chunk, nbytes)
        return buffer

    def free(self, buffer):
        chunk, _ = self._slices.pop(buffer.data_ptr())
        chunk.live -= 1
        if chunk.closing and chunk.live == 0:
            self._unpin_chunk(chunk)
            self._closing.remove(chunk)

    def recycle(self):
        """Hand out this phase's chunks again from their first byte.

        The caller must have freed every slice, and every queued copy that
        touched the arena must be ordered before the copies that will reuse it,
        for example because all of them run on one stream.
        """
        if self._slices:
            raise RuntimeError(f"{self.name} arena: recycle() with {len(self._slices)} live slice(s)")
        for chunk in self._chunks:
            chunk.next_offset = 0

    def _unpin_chunk(self, chunk):
        if self.registered:
            _unmap_registered(chunk.tensor)
        else:
            self._native.unpin(chunk.tensor)
        self.reserved_bytes -= chunk.nbytes
        chunk.tensor = None

    def release(self, synchronize=True):
        """End the phase: unpin all chunks, deferring those with live slices."""
        self._wait_reserved()
        if not self._chunks:
            return
        if synchronize:
            get_accelerator().synchronize()
        start = time.perf_counter()
        self.last_phase_bytes = sum(chunk.nbytes for chunk in self._chunks)
        for chunk in self._chunks:
            if chunk.live == 0:
                self._unpin_chunk(chunk)
            else:
                chunk.closing = True
                self._closing.append(chunk)
        self._chunks = []
        self.last_unpin_seconds = time.perf_counter() - start
        self.last_pin_seconds = self.phase_pin_seconds
        self.phase_pin_seconds = 0.0
        wait, self.last_wait_seconds = self.last_wait_seconds, 0.0
        placement = ""
        if self.registered:
            total = max(sum(self.phase_placement.values()), 1)
            nodes = ", ".join(f"node {n} {c / total:.0%}" for n, c in sorted(self.phase_placement.items(), key=str))
            placement = (f" (bound to node {self.numa_node}; sampled pages: {nodes}; "
                         f"{self.phase_unbound_chunks} chunk(s) not bound)")
            self.phase_placement.clear()
            self.phase_unbound_chunks = 0
        self.phases += 1
        print_rank_0(f"[superrl_pin] {self.name} arena phase {self.phases}: "
                     f"{self.last_phase_bytes / 2**30:.2f} GiB pinned in {self.last_pin_seconds:.3f} s "
                     f"(waited {wait:.3f} s for background pinning){placement}, "
                     f"unpinned in {self.last_unpin_seconds:.3f} s, "
                     f"{len(self._closing)} chunk(s) still held by live slices",
                     force=True)

    def stats(self):
        return {
            f"superrl_pin/{self.name}/phases": self.phases,
            f"superrl_pin/{self.name}/reserved_bytes": self.reserved_bytes,
            f"superrl_pin/{self.name}/peak_reserved_bytes": self.peak_reserved_bytes,
            f"superrl_pin/{self.name}/last_phase_bytes": self.last_phase_bytes,
            f"superrl_pin/{self.name}/last_pin_seconds": self.last_pin_seconds,
            f"superrl_pin/{self.name}/last_unpin_seconds": self.last_unpin_seconds,
        }
