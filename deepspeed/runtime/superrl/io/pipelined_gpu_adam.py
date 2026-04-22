# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""GPU-resident pipelined Adam over NVMe-resident optimizer state.

Paper sec. IV.C.2: ZeRO-Infinity swaps optimizer state to CPU and runs
CPU-Adam, which stalls on fine-grained NVMe<->CPU<->GPU hops and introduces
numerical drift vs. the non-offload path. SuperRL's replacement runs Adam
on the GPU with the same fused math as non-offloading training, using a
double-buffered NVMe->GPU pipeline (Fig. 8 in the paper):

    prefetch[i+1]  on  prefetch_engine  (background)
    compute  [i ]  on  GPU              (foreground)
    writeback[i]   on  writeback_engine (background)

Two specialised ``CoalescedNVMeEngine`` instances back the pipeline:

    self.read_engine    - non-blocking prefetch reads, deep queue
    self.write_engine   - non-blocking writeback writes, deep queue

This split is what makes the pipeline real. With a single engine, the
DeepNVMe ``handle.wait()`` semantics force a global drain that blocks
compute on unrelated future prefetches; with two engines we can wait on
read completion alone before consuming the next slot, while previous
writebacks proceed concurrently. End-of-step drains both.

Buffer placement:
- When ``read_engine.use_gds=True``  -> reads land directly in GPU.
- When ``read_engine.use_gds=False`` -> reads land in pinned host
  staging buffers and we add an explicit H2D copy before compute. This
  is the only correct way to use libaio with GPU compute.

Master weights live on NVMe (one extra file per param-key alongside m/v).
``param.data`` is the bf16 partitioned shard that ZeRO-3 expects; we
read fp32 master from NVMe, run AdamW with decoupled weight decay (to
match ``torch.optim.AdamW``), write the bf16 result into ``param.data``,
and write fp32 master/m/v back to NVMe.
"""

import os
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .nvme_engine import CoalescedNVMeEngine, IORequest


@dataclass
class _Chunk:
    """One pipeline unit: a contiguous slab of parameters."""

    chunk_id: int
    params: List[torch.nn.Parameter]
    numel: int
    spans: List[Tuple[int, int]]  # per-param (start, end) into the chunk's flat buffer
    keys: List[str]  # per-param state-file key


@dataclass
class _RingSlot:
    """A reusable triple of GPU buffers (master, m, v).

    When the read engine is libaio (no GDS), each slot also gets a
    matching triple of host-pinned staging buffers; the read targets
    those, and we H2D into the GPU buffers before compute.
    """

    slot_id: int
    master_buf: torch.Tensor
    m_buf: torch.Tensor
    v_buf: torch.Tensor
    host_master: Optional[torch.Tensor] = None
    host_m: Optional[torch.Tensor] = None
    host_v: Optional[torch.Tensor] = None


class SuperRLPipelinedGPUAdam(torch.optim.Optimizer):
    """Adam whose master/m/v live on NVMe and whose math runs on GPU."""

    KIND_MASTER = "master"
    KIND_M = "m"
    KIND_V = "v"

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        nvme_engine: CoalescedNVMeEngine,
        swap_folder: str,
        chunk_bytes: int = 128 * 1024 * 1024,
        ring_depth: int = 2,
        gpu_device: Optional[torch.device] = None,
        state_dtype: torch.dtype = torch.float32,
        write_engine: Optional[CoalescedNVMeEngine] = None,
    ):
        if nvme_engine is None:
            raise ValueError("SuperRLPipelinedGPUAdam requires an nvme_engine")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # The read engine is the user-supplied one (named generically
        # ``nvme_engine`` for backwards compat in stage3.py).
        self.read_engine = nvme_engine
        # We use a separate write engine so writeback waits do not block
        # prefetch and vice-versa. If the caller hasn't supplied one,
        # build a sibling with the same config; this is the case we hit
        # from stage3.py today.
        self.write_engine = write_engine or self._make_sibling_engine(nvme_engine)

        self.swap_folder = swap_folder
        self.chunk_bytes = int(chunk_bytes)
        self.ring_depth = max(2, int(ring_depth))
        self.gpu_device = gpu_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dtype = state_dtype
        # If the read engine is not GDS-capable we must stage in host
        # pinned buffers and H2D before compute.
        self._needs_host_staging = (
            self.read_engine._handle_kind != CoalescedNVMeEngine.HANDLE_GDS
        )

        os.makedirs(self.swap_folder, exist_ok=True)
        self._init_state_files_and_chunks()
        self._build_ring()

    @staticmethod
    def _make_sibling_engine(src: CoalescedNVMeEngine) -> CoalescedNVMeEngine:
        return CoalescedNVMeEngine(
            nvme_devices=src.nvme_devices,
            block_size=src.block_size,
            queue_depth=src.queue_depth,
            intra_op_parallelism=src.intra_op_parallelism,
            single_submit=src.single_submit,
            overlap_events=src.overlap_events,
            use_gds=src.use_gds,
        )

    # ------------------------------------------------------------------
    # State file & chunk construction
    # ------------------------------------------------------------------

    def _tensor_key(self, group_idx: int, param_idx: int) -> str:
        return f"g{group_idx}_p{param_idx}"

    def _state_path(self, key: str, kind: str) -> str:
        return os.path.join(self.swap_folder, f"{key}.{kind}.bin")

    def _init_state_files_and_chunks(self) -> None:
        """Group params into chunks and seed per-param state files.

        Master weights are seeded from ``param.data`` cast to fp32 on first
        construction; m/v are zeroed. State is written through the
        DeepNVMe write engine so the same code path (and pinned buffers)
        is used for both initialisation and every subsequent step.

        Across steps, master state is only ever read from / written to
        NVMe - never reconstructed from ``param.data`` (which may be the
        bf16 partitioned shard under ZeRO-3).
        """
        elem = torch.tensor([], dtype=self.state_dtype).element_size()
        chunk_elems = max(1, self.chunk_bytes // elem)

        self._chunks: List[_Chunk] = []
        cur_params: List[torch.nn.Parameter] = []
        cur_spans: List[Tuple[int, int]] = []
        cur_keys: List[str] = []
        cur_numel = 0
        chunk_id = 0

        seed_requests: List[IORequest] = []

        for gi, group in enumerate(self.param_groups):
            for pi, p in enumerate(group["params"]):
                key = self._tensor_key(gi, pi)
                self.state[p].setdefault("step", 0)
                self.state[p].setdefault("key", key)
                self.state[p].setdefault("group_idx", gi)
                n = p.numel()

                paths = {k: self._state_path(key, k) for k in (self.KIND_MASTER, self.KIND_M, self.KIND_V)}
                if not all(os.path.exists(pp) for pp in paths.values()):
                    master_buf = self.write_engine.allocate_host_buffer(n, self.state_dtype)
                    master_buf.copy_(p.detach().to(self.state_dtype).contiguous().view(-1).cpu())
                    m_buf = self.write_engine.allocate_host_buffer(n, self.state_dtype)
                    m_buf.zero_()
                    v_buf = self.write_engine.allocate_host_buffer(n, self.state_dtype)
                    v_buf.zero_()
                    seed_requests.append(IORequest(buffer=master_buf, path=paths[self.KIND_MASTER]))
                    seed_requests.append(IORequest(buffer=m_buf, path=paths[self.KIND_M]))
                    seed_requests.append(IORequest(buffer=v_buf, path=paths[self.KIND_V]))

                if cur_numel + n > chunk_elems and cur_numel > 0:
                    self._chunks.append(_Chunk(chunk_id, cur_params, cur_numel, cur_spans, cur_keys))
                    chunk_id += 1
                    cur_params, cur_spans, cur_keys, cur_numel = [], [], [], 0
                cur_spans.append((cur_numel, cur_numel + n))
                cur_keys.append(key)
                cur_params.append(p)
                cur_numel += n

        if cur_params:
            self._chunks.append(_Chunk(chunk_id, cur_params, cur_numel, cur_spans, cur_keys))

        self._max_chunk_numel = max((c.numel for c in self._chunks), default=0)

        if seed_requests:
            self.write_engine.submit_writes(seed_requests)
            self.write_engine.wait_all()

    # ------------------------------------------------------------------
    # GPU ring buffer
    # ------------------------------------------------------------------

    def _build_ring(self) -> None:
        if self._max_chunk_numel == 0:
            self._ring: List[_RingSlot] = []
            return

        self._ring = []
        for slot_id in range(self.ring_depth):
            master = self.read_engine.allocate_device_buffer(
                self._max_chunk_numel, self.state_dtype, device=self.gpu_device
            )
            m = self.read_engine.allocate_device_buffer(
                self._max_chunk_numel, self.state_dtype, device=self.gpu_device
            )
            v = self.read_engine.allocate_device_buffer(
                self._max_chunk_numel, self.state_dtype, device=self.gpu_device
            )
            slot = _RingSlot(slot_id=slot_id, master_buf=master, m_buf=m, v_buf=v)
            if self._needs_host_staging:
                slot.host_master = self.read_engine.allocate_host_buffer(
                    self._max_chunk_numel, self.state_dtype
                )
                slot.host_m = self.read_engine.allocate_host_buffer(
                    self._max_chunk_numel, self.state_dtype
                )
                slot.host_v = self.read_engine.allocate_host_buffer(
                    self._max_chunk_numel, self.state_dtype
                )
            self._ring.append(slot)

    # ------------------------------------------------------------------
    # Pipelined step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self._chunks:
            return loss

        free_slots: deque = deque(self._ring)
        in_flight: deque = deque()  # (chunk, slot)

        prefetch_idx = 0
        compute_idx = 0
        n = len(self._chunks)

        def issue_prefetch():
            nonlocal prefetch_idx
            if prefetch_idx >= n or not free_slots:
                return False
            slot = free_slots.popleft()
            chunk = self._chunks[prefetch_idx]
            self._submit_prefetch(chunk, slot)
            in_flight.append((chunk, slot))
            prefetch_idx += 1
            return True

        # Prime the ring: issue ring_depth prefetches up-front so the
        # pipeline has chunks ready when the first compute starts.
        while issue_prefetch():
            pass

        while compute_idx < n:
            chunk, slot = in_flight.popleft()
            # Wait only for the read queue to drain; write queue is
            # independent so previous writebacks keep streaming.
            self.read_engine.wait_all()
            if self._needs_host_staging:
                self._stage_h2d(chunk, slot)
            self._gpu_adam_compute(chunk, slot)
            self._submit_writeback(chunk, slot)
            free_slots.append(slot)
            compute_idx += 1
            # Issue the next prefetch *before* looping so it overlaps
            # with the ongoing GPU work and writeback.
            issue_prefetch()

        # Final drain: every writeback must hit storage before we return,
        # and every prefetch (none should remain, but be safe) too.
        self.read_engine.wait_all()
        self.write_engine.wait_all()

        for chunk in self._chunks:
            for p in chunk.params:
                self.state[p]["step"] = self.state[p].get("step", 0) + 1

        return loss

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _submit_prefetch(self, chunk: _Chunk, slot: _RingSlot) -> None:
        """Submit reads of (master, m, v) for every param in ``chunk`` into ``slot``.

        Reads target either the GPU buffers (GDS) or the host staging
        buffers (libaio). Writes always go through ``write_engine``.
        """
        if self._needs_host_staging:
            assert slot.host_master is not None
            master_dst = slot.host_master
            m_dst = slot.host_m
            v_dst = slot.host_v
        else:
            master_dst = slot.master_buf
            m_dst = slot.m_buf
            v_dst = slot.v_buf

        master_reqs: List[IORequest] = []
        m_reqs: List[IORequest] = []
        v_reqs: List[IORequest] = []
        for (start, end), key in zip(chunk.spans, chunk.keys):
            master_reqs.append(IORequest(
                buffer=master_dst[start:end], path=self._state_path(key, self.KIND_MASTER), group=f"chunk{chunk.chunk_id}_master"))
            m_reqs.append(IORequest(
                buffer=m_dst[start:end], path=self._state_path(key, self.KIND_M), group=f"chunk{chunk.chunk_id}_m"))
            v_reqs.append(IORequest(
                buffer=v_dst[start:end], path=self._state_path(key, self.KIND_V), group=f"chunk{chunk.chunk_id}_v"))
        self.read_engine.submit_reads(master_reqs)
        self.read_engine.submit_reads(m_reqs)
        self.read_engine.submit_reads(v_reqs)

    def _stage_h2d(self, chunk: _Chunk, slot: _RingSlot) -> None:
        """Copy a libaio-prefetched chunk from host pinned to GPU buffers."""
        if self.gpu_device.type != "cuda":
            slot.master_buf.narrow(0, 0, chunk.numel).copy_(slot.host_master.narrow(0, 0, chunk.numel))
            slot.m_buf.narrow(0, 0, chunk.numel).copy_(slot.host_m.narrow(0, 0, chunk.numel))
            slot.v_buf.narrow(0, 0, chunk.numel).copy_(slot.host_v.narrow(0, 0, chunk.numel))
            return
        slot.master_buf.narrow(0, 0, chunk.numel).copy_(
            slot.host_master.narrow(0, 0, chunk.numel), non_blocking=True
        )
        slot.m_buf.narrow(0, 0, chunk.numel).copy_(
            slot.host_m.narrow(0, 0, chunk.numel), non_blocking=True
        )
        slot.v_buf.narrow(0, 0, chunk.numel).copy_(
            slot.host_v.narrow(0, 0, chunk.numel), non_blocking=True
        )
        # Issue compute on the same stream, so the compute kernel naturally
        # serialises after the H2D copies.

    def _submit_writeback(self, chunk: _Chunk, slot: _RingSlot) -> None:
        """Submit writes of (master, m, v) for ``chunk`` from ``slot``.

        For libaio we must D2H copy back into host pinned buffers before
        submitting (libaio cannot accept GPU buffers); for GDS the write
        engine accepts the GPU buffer directly.
        """
        if self._needs_host_staging:
            assert slot.host_master is not None
            slot.host_master.narrow(0, 0, chunk.numel).copy_(
                slot.master_buf.narrow(0, 0, chunk.numel).cpu()
            )
            slot.host_m.narrow(0, 0, chunk.numel).copy_(
                slot.m_buf.narrow(0, 0, chunk.numel).cpu()
            )
            slot.host_v.narrow(0, 0, chunk.numel).copy_(
                slot.v_buf.narrow(0, 0, chunk.numel).cpu()
            )
            master_src = slot.host_master
            m_src = slot.host_m
            v_src = slot.host_v
        else:
            master_src = slot.master_buf
            m_src = slot.m_buf
            v_src = slot.v_buf

        master_reqs: List[IORequest] = []
        m_reqs: List[IORequest] = []
        v_reqs: List[IORequest] = []
        for (start, end), key in zip(chunk.spans, chunk.keys):
            master_reqs.append(IORequest(
                buffer=master_src[start:end], path=self._state_path(key, self.KIND_MASTER), group=f"chunk{chunk.chunk_id}_master"))
            m_reqs.append(IORequest(
                buffer=m_src[start:end], path=self._state_path(key, self.KIND_M), group=f"chunk{chunk.chunk_id}_m"))
            v_reqs.append(IORequest(
                buffer=v_src[start:end], path=self._state_path(key, self.KIND_V), group=f"chunk{chunk.chunk_id}_v"))
        self.write_engine.submit_writes(master_reqs)
        self.write_engine.submit_writes(m_reqs)
        self.write_engine.submit_writes(v_reqs)

    def _gpu_adam_compute(self, chunk: _Chunk, slot: _RingSlot) -> None:
        """Run AdamW on every param in ``chunk`` using ``slot`` buffers.

        Decoupled weight decay matches ``torch.optim.AdamW``. Master/m/v
        are updated in place inside the slot; the bf16 (or whatever
        ``param.dtype``) shard in ``param.data`` is overwritten with the
        new master cast back down so subsequent forward passes see the
        update.
        """
        for p, (start, end) in zip(chunk.params, chunk.spans):
            if p.grad is None:
                continue
            gi = self.state[p].get("group_idx", 0)
            group = self.param_groups[gi]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            step = self.state[p].get("step", 0) + 1
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step

            m = slot.m_buf[start:end]
            v = slot.v_buf[start:end]
            master = slot.master_buf[start:end]
            grad = p.grad.detach().view(-1).to(self.state_dtype)

            if wd != 0.0:
                master.mul_(1.0 - lr * wd)

            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            denom = (v / bc2).sqrt_().add_(eps)
            master.addcdiv_(m, denom, value=-(lr / bc1))

            p.data.view(-1).copy_(master.to(p.dtype))

    # ------------------------------------------------------------------

    def stats(self) -> dict:
        s = self.read_engine.stats()
        # Merge write engine telemetry under a `_w` suffix to keep keys unique.
        for k, v in self.write_engine.stats().items():
            s[k.replace("/", "/w_", 1)] = v
        s["superrl_io/optimizer_chunks"] = len(self._chunks)
        s["superrl_io/ring_depth"] = self.ring_depth
        s["superrl_io/chunk_bytes"] = self.chunk_bytes
        s["superrl_io/host_staging"] = self._needs_host_staging
        return s
