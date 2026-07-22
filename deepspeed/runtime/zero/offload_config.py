# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import Enum
import os
from pathlib import Path
from pydantic import Field, model_validator
from typing import List, Optional

from deepspeed.runtime.config_utils import DeepSpeedConfigModel, pp_int


ADAM_OPTIMIZER_STATE_BUFFER_COUNT = 4
NVME_PATH_RESOLVED_FROM_PER_LOCAL_RANK = "_nvme_path_resolved_from_per_local_rank"


def local_rank_from_env():
    for name in ("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 0


def has_nvme_path(offload_config):
    if offload_config is None:
        return False
    return getattr(offload_config, "nvme_path", None) is not None or bool(
        getattr(offload_config, "nvme_path_per_local_rank", None))


def resolve_nvme_path(offload_config, local_rank=None):
    nvme_path = getattr(offload_config, "nvme_path", None)
    if getattr(offload_config, NVME_PATH_RESOLVED_FROM_PER_LOCAL_RANK, False) and nvme_path is not None:
        return nvme_path

    per_local_rank = getattr(offload_config, "nvme_path_per_local_rank", None)
    if per_local_rank:
        if local_rank is None:
            local_rank = local_rank_from_env()
        if local_rank < 0 or local_rank >= len(per_local_rank):
            raise ValueError(
                f"LOCAL_RANK={local_rank} cannot use nvme_path_per_local_rank with {len(per_local_rank)} path(s).")
        return per_local_rank[local_rank]
    return nvme_path


def set_nvme_path_from_per_local_rank(offload_config, local_rank=None):
    per_local_rank = getattr(offload_config, "nvme_path_per_local_rank", None)
    if not per_local_rank:
        return offload_config

    if local_rank is None:
        local_rank = local_rank_from_env()
    if local_rank < 0 or local_rank >= len(per_local_rank):
        raise ValueError(
            f"LOCAL_RANK={local_rank} cannot use nvme_path_per_local_rank with {len(per_local_rank)} path(s).")

    offload_config.__dict__["nvme_path"] = per_local_rank[local_rank]
    offload_config.__dict__[NVME_PATH_RESOLVED_FROM_PER_LOCAL_RANK] = True
    return offload_config


class OffloadDeviceEnum(str, Enum):
    """ Enum for valid offload devices """
    none = "none"
    cpu = "cpu"
    nvme = "nvme"


class DeepSpeedZeroOffloadParamConfig(DeepSpeedConfigModel):
    """ Set options for parameter offload. Valid only with stage 3. """

    device: OffloadDeviceEnum = "none"
    """
    Device memory to offload model parameters. Supported options are `cpu` and
    `nvme`.
    """

    nvme_path: Optional[Path] = None
    """ Filesystem path for NVMe device for parameter offloading. """

    nvme_path_per_local_rank: Optional[List[Path]] = None
    """ Filesystem paths for NVMe parameter offload, indexed by LOCAL_RANK. """

    buffer_count: int = Field(5, ge=0)
    """ Number of buffers in buffer pool for parameter offloading to NVMe. """

    buffer_size: int = Field(pp_int(1e8), ge=0)
    """ Size of buffers in buffer pool for parameter offloading to NVMe. """

    max_in_cpu: int = Field(pp_int(1e9), ge=0)
    """
    Number of parameter elements to maintain in CPU memory when offloading to
    NVMe is enabled.
    """

    pin_memory: bool = False
    """
    Offload to page-locked CPU memory. This could boost throughput at the cost
    of extra memory overhead.
    """

    @model_validator(mode="after")
    def set_rank_local_nvme_path(self):
        return set_nvme_path_from_per_local_rank(self)


class DeepSpeedZeroOffloadOptimizerConfig(DeepSpeedConfigModel):
    """ Set options for optimizer offload. Valid with stage 1, 2, and 3. """

    device: OffloadDeviceEnum = "none"
    """
    Device memory to offload optimizer state. Supported options are `cpu` and
    `nvme`. Optimizer computation is offload to CPU regardless of device option.
    """

    nvme_path: Optional[Path] = None
    """ Filesystem path for NVMe device for optimizer state offloading. """

    nvme_path_per_local_rank: Optional[List[Path]] = None
    """ Filesystem paths for NVMe optimizer offload, indexed by LOCAL_RANK. """

    buffer_count: int = Field(4, ge=0)
    """
    Number of buffers in buffer pool for optimizer state offloading to NVMe.
    This should be at least the number of states maintained per parameter by
    the optimizer. For example, Adam optimizer has 4 states (parameter,
    gradient, momentum, and variance).
    """

    buffer_size: int = Field(pp_int(1e8), ge=0)
    """
    Number of elements per staging buffer for optimizer state offloading to NVMe.
    Pipeline mode uses this as the chunk size for temporary pinned write staging
    buffers.
    """

    pin_memory: bool = False
    """
    Offload to page-locked CPU memory. This could boost throughput at the cost
    of extra memory overhead.
    """

    pipeline_read: bool = False
    """
    For tile-based optimizer step processing, overlap read of next tile with
    computation of current tile. Used in ZeRO-Infinity.
    """

    pipeline_write: bool = False
    """
    For tile-based optimizer step processing, overlap write of previous tile
    with computation of current tile.
    """

    fast_init: bool = False
    """ Enable fast optimizer initialization when offloading to NVMe. """

    ratio: float = Field(1.0, ge=0.0, le=1.0)
    """ Percentage of offloaded optimizer states to CPU Adam. Only valid with ZeRO Stage 3."""

    super_offload: bool = False
    """ Enable high performance CPU offloading for Superchips. Only valid with ZeRO Stage 3."""

    cpuadam_cores_perc: float = Field(0.8, ge=0.0, le=1.0)
    """ Percentage of CPU cores to use for CPU Adam. Only valid with ZeRO Stage 3 and super_offload=True."""

    @model_validator(mode="after")
    def set_pipeline(self):
        set_nvme_path_from_per_local_rank(self)
        pipeline = self.pipeline_read or self.pipeline_write
        self.__dict__["pipeline"] = pipeline
        if self.device == OffloadDeviceEnum.nvme and pipeline:
            min_buffer_count = self._minimum_pipeline_buffer_count()
            if self.buffer_count < min_buffer_count:
                raise ValueError(
                    "NVMe optimizer offload with pipeline_read="
                    f"{self.pipeline_read} and pipeline_write={self.pipeline_write} "
                    f"requires buffer_count >= {min_buffer_count}; got {self.buffer_count}. "
                    "Pipelined swapping can concurrently hold current swap-in buffers, "
                    "next async swap-in buffers, previous async swap-out buffers, and "
                    "write staging buffers. Increase buffer_count or disable optimizer "
                    "offload pipelining.")
            if self.buffer_size <= 0:
                raise ValueError("NVMe optimizer offload pipelining requires buffer_size > 0.")
        return self

    def _minimum_pipeline_buffer_count(self) -> int:
        # Adam keeps parameter, gradient, momentum, and variance state in the
        # optimizer swap path. Pipeline mode may hold multiple such groups at
        # once, so the non-pipelined minimum is not sufficient.
        sync_swap_in = ADAM_OPTIMIZER_STATE_BUFFER_COUNT
        next_async_swap_in = ADAM_OPTIMIZER_STATE_BUFFER_COUNT if self.pipeline_read else 0
        previous_async_swap_out = 2 * ADAM_OPTIMIZER_STATE_BUFFER_COUNT if self.pipeline_write else 0
        write_staging = ADAM_OPTIMIZER_STATE_BUFFER_COUNT
        return max(sync_swap_in + next_async_swap_in + write_staging,
                   previous_async_swap_out + sync_swap_in + next_async_swap_in)


class OffloadStateTypeEnum(str, Enum):
    """ Enum for internal buffer types """
    optim_states = "optim_states"
    hp_params = "hp_params"
    lp_params = "lp_params"
    lp_grads = "lp_grads"
    contiguous_grad_buffer = "contiguous_grad_buffer"
