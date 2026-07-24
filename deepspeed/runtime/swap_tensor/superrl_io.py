# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Runtime helpers for SuperRL-IO.

The public SuperRL-IO configuration is intentionally a single boolean. This
module keeps the operational checks and defaults inside DeepSpeed so training
configs do not grow hardware-specific tuning fields.
"""

import os
import copy
import subprocess
import sys
import textwrap
from dataclasses import dataclass

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.swap_tensor.constants import AIO_BLOCK_SIZE, AIO_INTRA_OP_PARALLELISM, AIO_OVERLAP_EVENTS, \
    AIO_QUEUE_DEPTH, AIO_SINGLE_SUBMIT, AIO_THREAD_COUNT
from deepspeed.runtime.zero.offload_config import resolve_nvme_path

SUPERRL_IO_MIN_OPTIMIZER_BUFFER_COUNT = 16
SUPERRL_IO_SYNC_MIN_OPTIMIZER_BUFFER_COUNT = 5
SUPERRL_IO_BUFFERS_PER_PREFETCH = 4


@dataclass(frozen=True)
class GDSProbeResult:
    ok: bool
    message: str
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def _enum_value(value):
    return value.value if hasattr(value, "value") else value


def _tail(text, limit=2000):
    if not text:
        return ""
    return text[-limit:]


def _with_parallelism(aio_config, parallelism):
    if parallelism is None:
        return aio_config
    config = copy.copy(aio_config)
    parallelism = max(1, int(parallelism))
    config[AIO_INTRA_OP_PARALLELISM] = parallelism
    config[AIO_THREAD_COUNT] = parallelism
    return config


def probe_gds_path(nvme_path, aio_config, timeout_sec=None):
    """Return whether cuFile can register and move data for ``nvme_path``.

    DeepSpeed's current GDS extension exits the process on some cuFile failures.
    The probe therefore runs in a subprocess and treats any non-zero return code
    as a clean failure result for the parent training process.
    """
    accelerator = get_accelerator()
    if accelerator.device_name() != "cuda" or not accelerator.is_available():
        return GDSProbeResult(ok=False, message="SuperRL-IO requires an available CUDA accelerator.")
    device_index = int(accelerator.current_device())

    nvme_path = os.fspath(nvme_path)
    os.makedirs(nvme_path, exist_ok=True)

    block_size = int(aio_config[AIO_BLOCK_SIZE])
    queue_depth = int(aio_config[AIO_QUEUE_DEPTH])
    single_submit = bool(aio_config[AIO_SINGLE_SUBMIT])
    overlap_events = bool(aio_config[AIO_OVERLAP_EVENTS])
    intra_op_parallelism = int(aio_config[AIO_INTRA_OP_PARALLELISM])
    probe_bytes = max(block_size * max(1, intra_op_parallelism), 1024 * 1024)

    probe_script = textwrap.dedent("""
        import os
        import sys

        import torch

        from deepspeed.accelerator import get_accelerator
        from deepspeed.ops.op_builder import GDSBuilder

        path = sys.argv[1]
        block_size = int(sys.argv[2])
        queue_depth = int(sys.argv[3])
        single_submit = sys.argv[4] == "1"
        overlap_events = sys.argv[5] == "1"
        intra_op_parallelism = int(sys.argv[6])
        probe_bytes = int(sys.argv[7])
        device_index = int(sys.argv[8])

        os.makedirs(path, exist_ok=True)
        probe_file = os.path.join(path, f".deepspeed_superrl_io_gds_probe_{os.getpid()}.bin")
        handle = None
        buffer = None
        try:
            torch.cuda.set_device(device_index)
            handle = GDSBuilder().load(verbose=False).gds_handle(
                block_size=block_size,
                queue_depth=queue_depth,
                single_submit=single_submit,
                overlap_events=overlap_events,
                intra_op_parallelism=intra_op_parallelism,
            )
            buffer = torch.empty(probe_bytes, dtype=torch.uint8, device=f"cuda:{device_index}")
            buffer.zero_()
            handle.pin_device_tensor(buffer)

            write_status = handle.async_pwrite(buffer, probe_file, 0)
            if write_status != 0:
                raise RuntimeError(f"async_pwrite returned {write_status}")
            write_wait = handle.wait()
            if write_wait != 1:
                raise RuntimeError(f"write wait returned {write_wait}, expected 1")

            buffer.fill_(1)
            read_status = handle.async_pread(buffer, probe_file, 0)
            if read_status != 0:
                raise RuntimeError(f"async_pread returned {read_status}")
            read_wait = handle.wait()
            if read_wait != 1:
                raise RuntimeError(f"read wait returned {read_wait}, expected 1")

            if int(buffer.sum().item()) != 0:
                raise RuntimeError("readback validation failed")

            print("SuperRL-IO GDS probe passed")
        finally:
            if handle is not None and buffer is not None:
                try:
                    handle.unpin_device_tensor(buffer)
                except Exception:
                    pass
            try:
                os.remove(probe_file)
            except FileNotFoundError:
                pass
    """)

    cmd = [
        sys.executable,
        "-c",
        probe_script,
        nvme_path,
        str(block_size),
        str(queue_depth),
        "1" if single_submit else "0",
        "1" if overlap_events else "0",
        str(intra_op_parallelism),
        str(probe_bytes),
        str(device_index),
    ]

    if timeout_sec is None:
        timeout_sec = int(os.environ.get("SUPERRL_GDS_PROBE_TIMEOUT_SEC", "120"))

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired as exc:
        details = " ".join(part for part in [_tail(exc.stdout), _tail(exc.stderr)] if part).strip()
        return GDSProbeResult(ok=False,
                              message=f"GDS probe timed out after {timeout_sec}s for {nvme_path}: {details}")

    if completed.returncode == 0:
        return GDSProbeResult(ok=True,
                              message=f"GDS probe passed for {nvme_path}",
                              returncode=completed.returncode,
                              stdout=completed.stdout,
                              stderr=completed.stderr)

    details = " ".join(part for part in [_tail(completed.stdout), _tail(completed.stderr)] if part).strip()
    return GDSProbeResult(ok=False,
                          message=f"GDS probe failed for {nvme_path} with exit code {completed.returncode}: {details}",
                          returncode=completed.returncode,
                          stdout=completed.stdout,
                          stderr=completed.stderr)


def ensure_superrl_io_gds_ready(superrl_io_config, offload_optimizer_config, aio_config, probe_fn=probe_gds_path):
    if not getattr(superrl_io_config, "enabled", False):
        return

    if offload_optimizer_config is None:
        raise ValueError("SuperRL-IO requires zero_optimization.offload_optimizer with device='nvme'.")

    optimizer_device = _enum_value(getattr(offload_optimizer_config, "device", None))
    if optimizer_device != "nvme":
        raise ValueError(
            f"SuperRL-IO requires zero_optimization.offload_optimizer.device='nvme', got {optimizer_device!r}.")

    nvme_path = resolve_nvme_path(offload_optimizer_config)
    if nvme_path is None:
        raise ValueError("SuperRL-IO requires zero_optimization.offload_optimizer.nvme_path or "
                         "nvme_path_per_local_rank.")

    probe_aio_config = _with_parallelism(aio_config, getattr(superrl_io_config, "read_thread_count", None))
    result = probe_fn(nvme_path, probe_aio_config)
    if not result.ok:
        raise RuntimeError(
            "SuperRL-IO was enabled, but real GPUDirect Storage is not usable on the optimizer NVMe path. "
            "Install/load nvidia-fs and verify gdscheck -f on the target path before retrying. "
            f"{result.message}")


def make_superrl_io_swap_config(swap_config, superrl_io_config=None):
    """Return an internal optimizer swap config for SuperRL-IO.

    The user-facing control remains ``superrl_io: true``. The hidden defaults
    below enable read/write pipelining by default. Explicitly disabling both
    pipeline directions selects a synchronous five-buffer GDS path for
    host-memory-constrained runs.
    """
    config = copy.copy(swap_config)
    pipeline_read = True if superrl_io_config is None else bool(getattr(superrl_io_config, "pipeline_read", True))
    pipeline_write = True if superrl_io_config is None else bool(getattr(superrl_io_config, "pipeline_write", True))
    config.__dict__["pipeline_read"] = pipeline_read
    config.__dict__["pipeline_write"] = pipeline_write
    config.__dict__["pipeline"] = pipeline_read or pipeline_write
    read_thread_count = None if superrl_io_config is None else getattr(superrl_io_config, "read_thread_count", None)
    write_thread_count = None if superrl_io_config is None else getattr(superrl_io_config, "write_thread_count", None)
    prefetch_depth = 1 if superrl_io_config is None else max(1, int(getattr(superrl_io_config, "prefetch_depth", 1)))
    if not config.pipeline:
        prefetch_depth = 1
    config.__dict__["gds_prefetch_depth"] = prefetch_depth
    if read_thread_count is not None:
        config.__dict__["gds_read_intra_op_parallelism"] = int(read_thread_count)
    if write_thread_count is not None:
        config.__dict__["gds_write_intra_op_parallelism"] = int(write_thread_count)
    if config.pipeline:
        min_buffer_count = SUPERRL_IO_MIN_OPTIMIZER_BUFFER_COUNT + \
            (prefetch_depth - 1) * SUPERRL_IO_BUFFERS_PER_PREFETCH
    else:
        min_buffer_count = SUPERRL_IO_SYNC_MIN_OPTIMIZER_BUFFER_COUNT
    config.__dict__["buffer_count"] = max(int(getattr(config, "buffer_count", 0)), min_buffer_count)
    return config
