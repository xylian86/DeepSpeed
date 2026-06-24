# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import Enum
from pathlib import Path
from pydantic import Field, model_validator
from typing import Optional

from deepspeed.runtime.config_utils import DeepSpeedConfigModel, pp_int


ADAM_OPTIMIZER_STATE_BUFFER_COUNT = 4


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


class DeepSpeedZeroOffloadOptimizerConfig(DeepSpeedConfigModel):
    """ Set options for optimizer offload. Valid with stage 1, 2, and 3. """

    device: OffloadDeviceEnum = "none"
    """
    Device memory to offload optimizer state. Supported options are `cpu` and
    `nvme`. Optimizer computation is offload to CPU regardless of device option.
    """

    nvme_path: Optional[Path] = None
    """ Filesystem path for NVMe device for optimizer state offloading. """

    buffer_count: int = Field(4, ge=0)
    """
    Number of buffers in buffer pool for optimizer state offloading to NVMe.
    This should be at least the number of states maintained per parameter by
    the optimizer. For example, Adam optimizer has 4 states (parameter,
    gradient, momentum, and variance).
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
