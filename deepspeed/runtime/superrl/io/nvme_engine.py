# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""DeepNVMe-backed NVMe engine for SuperRL-IO.

DeepNVMe (``aio_handle`` for libaio, ``gds_handle`` for GPUDirect Storage)
already gives us non-blocking deep-queue submission, ``intra_op_parallelism``
inside a single IO, and the cuFile fast path. SuperRL-IO's contribution
(paper sec. IV.C.1) on top of that is:

1. **Coalesced large transfers.** A planner pass merges adjacent same-storage
   shards in a submission batch into one large pread/pwrite, eliminating
   per-shard DMA setup cost.
2. **Per-device routing.** One DeepNVMe handle per physical NVMe mount in
   ``nvme_devices``; reads/writes are dispatched by file basename so that
   multiple NVMe SSDs stripe naturally without requiring mdadm.
3. **Statistics.** Coalescing ratio, bytes moved, and handle kind are exposed
   to the trainer's ``log_dict``.

The pipelined GPU Adam (paper sec. IV.C.2) lives in
``pipelined_gpu_adam.py`` and uses this engine for IO.
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch


@dataclass
class IORequest:
    """One pending IO against the NVMe engine.

    Parameters
    ----------
    buffer:
        DMA-pinned tensor (host-pinned for ``aio_handle``, device-pinned for
        ``gds_handle``). Use the engine's ``allocate_host_buffer`` /
        ``allocate_device_buffer`` to get the right kind.
    path:
        Absolute file path. Used both as the IO target and for stripe routing.
    offset:
        Byte offset inside ``path``. ``0`` is the common case when each tensor
        lives in its own file (DeepSpeed's existing swapper convention).
    group:
        Optional tag used by the coalescer. Two requests merge only if their
        ``group`` matches.
    """

    buffer: torch.Tensor
    path: str
    offset: int = 0
    group: Optional[str] = None
    _stripe: int = field(default=-1, repr=False)


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _storage_data_ptr(t: torch.Tensor) -> int:
    storage = t.untyped_storage() if hasattr(t, "untyped_storage") else t.storage()
    return storage.data_ptr()


class CoalescedNVMeEngine:
    """Front-end to one DeepNVMe handle per NVMe device.

    Example::

        engine = CoalescedNVMeEngine.from_config(io_config)
        buf = engine.allocate_host_buffer(numel=N, dtype=torch.float32)
        engine.submit_reads([IORequest(buffer=buf, path=path, offset=0)])
        engine.wait_all()
    """

    HANDLE_AIO = "aio"
    HANDLE_GDS = "gds"

    def __init__(
        self,
        nvme_devices: Sequence[str],
        block_size: int,
        queue_depth: int,
        intra_op_parallelism: int,
        single_submit: bool,
        overlap_events: bool,
        use_gds: bool,
    ) -> None:
        self.nvme_devices = list(nvme_devices) or [""]
        self.block_size = int(block_size)
        self.queue_depth = int(queue_depth)
        self.intra_op_parallelism = int(intra_op_parallelism)
        self.single_submit = bool(single_submit)
        self.overlap_events = bool(overlap_events)
        self.use_gds = bool(use_gds)

        self._handle_kind, self._handle_factory = self._resolve_handle_factory()
        self._handles = [self._handle_factory() for _ in self.nvme_devices]
        self._pending = [0 for _ in self._handles]

        self.total_bytes_read = 0
        self.total_bytes_written = 0
        self.num_raw_requests = 0
        self.num_submitted_requests = 0

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, io_config) -> "CoalescedNVMeEngine":
        return cls(
            nvme_devices=io_config.nvme_devices,
            block_size=io_config.block_size,
            queue_depth=io_config.queue_depth,
            intra_op_parallelism=io_config.intra_op_parallelism,
            single_submit=io_config.single_submit,
            overlap_events=io_config.overlap_events,
            use_gds=io_config.use_gds,
        )

    def _resolve_handle_factory(self):
        """Return (handle_kind, factory) using DeepNVMe builders.

        Raises a clear error if libaio (and, when requested, GDS) is not
        available - the artifact targets GH200 where both should be present;
        silently degrading would hide configuration mistakes.
        """
        try:
            from deepspeed.ops.op_builder import AsyncIOBuilder  # type: ignore
        except Exception as exc:  # pragma: no cover - import error path
            raise RuntimeError(
                "SuperRL-IO requires DeepSpeed's async_io op (DeepNVMe). "
                "Build/install DeepSpeed with libaio-dev installed; verify "
                "with `ds_report`."
            ) from exc

        aio_builder = AsyncIOBuilder()
        if not aio_builder.is_compatible():
            raise RuntimeError(
                "DeepNVMe async_io op is not compatible in this environment. "
                "Install libaio-dev (Ubuntu: `apt install libaio-dev`) and "
                "rebuild DeepSpeed; verify with `ds_report`."
            )

        if self.use_gds:
            try:
                from deepspeed.ops.op_builder import GDSBuilder  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "SuperRL-IO use_gds=true requires DeepSpeed's gds op. "
                    "Install nvidia-fs and rebuild DeepSpeed."
                ) from exc

            gds_builder = GDSBuilder()
            if not gds_builder.is_compatible():
                raise RuntimeError(
                    "DeepNVMe gds op is not compatible. Install nvidia-fs "
                    "(GPUDirect Storage) and rebuild DeepSpeed; verify with "
                    "`ds_report`."
                )
            gds_op = gds_builder.load()

            def make_gds():
                return gds_op.gds_handle(
                    block_size=self.block_size,
                    queue_depth=self.queue_depth,
                    single_submit=self.single_submit,
                    overlap_events=self.overlap_events,
                    intra_op_parallelism=self.intra_op_parallelism,
                )

            return self.HANDLE_GDS, make_gds

        aio_op = aio_builder.load()

        def make_aio():
            return aio_op.aio_handle(
                block_size=self.block_size,
                queue_depth=self.queue_depth,
                single_submit=self.single_submit,
                overlap_events=self.overlap_events,
                intra_op_parallelism=self.intra_op_parallelism,
            )

        return self.HANDLE_AIO, make_aio

    # ------------------------------------------------------------------
    # Pinned buffer allocation (DMA-capable)
    # ------------------------------------------------------------------

    def allocate_host_buffer(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """Allocate a host-pinned tensor suitable for ``async_pread``/``pwrite``.

        Uses the aio handle's ``new_cpu_locked_tensor`` when possible; falls
        back to ``torch.empty(...).pin_memory()`` if the handle does not
        export the helper (older DeepSpeed builds).
        """
        handle = self._handles[0]
        new_locked = getattr(handle, "new_cpu_locked_tensor", None)
        if new_locked is not None:
            tensor = new_locked(numel, torch.empty(0, dtype=dtype))
            return tensor.view(dtype)
        empty = torch.empty(numel, dtype=dtype)
        try:
            return empty.pin_memory()
        except (RuntimeError, NotImplementedError):
            # No CUDA host pinner (MacOS/MPS, ROCm without HIP, CPU-only).
            return empty

    def free_host_buffer(self, tensor: torch.Tensor) -> None:
        handle = self._handles[0]
        free_locked = getattr(handle, "free_cpu_locked_tensor", None)
        if free_locked is not None:
            free_locked(tensor)

    def allocate_device_buffer(self, numel: int, dtype: torch.dtype, device=None) -> torch.Tensor:
        """Allocate a CUDA tensor and pin it for GDS direct DMA when available."""
        device = device or torch.device("cuda")
        tensor = torch.empty(numel, dtype=dtype, device=device)
        if self._handle_kind == self.HANDLE_GDS:
            for handle in self._handles:
                pin = getattr(handle, "pin_device_tensor", None)
                if pin is not None:
                    pin(tensor)
                    break
        return tensor

    def free_device_buffer(self, tensor: torch.Tensor) -> None:
        if self._handle_kind == self.HANDLE_GDS:
            for handle in self._handles:
                unpin = getattr(handle, "unpin_device_tensor", None)
                if unpin is not None:
                    try:
                        unpin(tensor)
                    except Exception:
                        pass
                    break

    # ------------------------------------------------------------------
    # Routing + coalescing planner
    # ------------------------------------------------------------------

    def _stripe_for(self, path: str) -> int:
        """Stable basename hash routing - independent of mount-prefix layout."""
        if len(self._handles) == 1:
            return 0
        digest = hashlib.blake2b(os.path.basename(path).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % len(self._handles)

    def _can_merge(self, a: IORequest, b: IORequest) -> bool:
        if a.path != b.path or a.group != b.group:
            return False
        a_end_offset = a.offset + _nbytes(a.buffer)
        if a_end_offset != b.offset:
            return False
        # Buffers must live in the same storage and be byte-contiguous.
        if a.buffer.dtype != b.buffer.dtype:
            return False
        if _storage_data_ptr(a.buffer) != _storage_data_ptr(b.buffer):
            return False
        a_end_ptr = a.buffer.data_ptr() + _nbytes(a.buffer)
        return a_end_ptr == b.buffer.data_ptr()

    def _coalesce(self, requests: Sequence[IORequest]) -> List[IORequest]:
        """Fold contiguous same-(path,group,storage) requests into one IO.

        Returns the list of merged IORequests actually handed to DeepNVMe.
        Coalesced requests are zero-copy: their ``buffer`` is a slice of the
        same parent storage that the originals were sliced from.
        """
        if not requests:
            return []
        self.num_raw_requests += len(requests)
        merged: List[IORequest] = []
        for req in requests:
            if merged and self._can_merge(merged[-1], req):
                prev = merged[-1]
                # Build a single tensor view spanning prev+req over the shared storage.
                # We reinterpret the shared underlying storage as a 1-D byte tensor
                # then take the slice covering prev.start .. req.end. This avoids
                # any per-element copy.
                storage = prev.buffer.untyped_storage() if hasattr(prev.buffer, "untyped_storage") else prev.buffer.storage()
                base = torch.tensor([], dtype=torch.uint8).set_(storage)  # 1-D byte view
                base_ptr = storage.data_ptr()
                start_byte = prev.buffer.data_ptr() - base_ptr
                end_byte = req.buffer.data_ptr() + _nbytes(req.buffer) - base_ptr
                merged_bytes = base[start_byte:end_byte]
                merged_dtype_view = merged_bytes.view(prev.buffer.dtype)
                merged_req = IORequest(
                    buffer=merged_dtype_view,
                    path=prev.path,
                    offset=prev.offset,
                    group=prev.group,
                )
                merged[-1] = merged_req
            else:
                merged.append(req)
        self.num_submitted_requests += len(merged)
        return merged

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def _submit(self, requests: Sequence[IORequest], read: bool) -> None:
        if not requests:
            return
        for merged in self._coalesce(requests):
            stripe = self._stripe_for(merged.path)
            handle = self._handles[stripe]
            nbytes = _nbytes(merged.buffer)
            if read:
                handle.async_pread(merged.buffer, merged.path)
                self.total_bytes_read += nbytes
            else:
                handle.async_pwrite(merged.buffer, merged.path)
                self.total_bytes_written += nbytes
            self._pending[stripe] += 1

    def submit_reads(self, requests: Sequence[IORequest]) -> None:
        self._submit(requests, read=True)

    def submit_writes(self, requests: Sequence[IORequest]) -> None:
        self._submit(requests, read=False)

    def wait_all(self) -> None:
        """Drain every stripe's queue. Must be called before touching result buffers."""
        for i, handle in enumerate(self._handles):
            if self._pending[i] == 0:
                continue
            handle.wait()
            self._pending[i] = 0

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        coalescing_ratio = 0.0
        if self.num_raw_requests:
            coalescing_ratio = 1.0 - (self.num_submitted_requests / self.num_raw_requests)
        return {
            "superrl_io/handle_kind": self._handle_kind,
            "superrl_io/stripes": len(self._handles),
            "superrl_io/bytes_read": self.total_bytes_read,
            "superrl_io/bytes_written": self.total_bytes_written,
            "superrl_io/raw_requests": self.num_raw_requests,
            "superrl_io/submitted_requests": self.num_submitted_requests,
            "superrl_io/coalescing_ratio": coalescing_ratio,
        }
