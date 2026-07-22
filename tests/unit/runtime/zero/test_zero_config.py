# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
from pathlib import Path
from pydantic import ValidationError

from deepspeed.runtime.config import get_superrl_cache_config, get_superrl_io_config, get_superrl_sync_config
from deepspeed.runtime.swap_tensor.aio_config import get_aio_config
from deepspeed.runtime.swap_tensor.constants import AIO_IO_BACKEND, AIO_IO_BACKEND_ASYNC, AIO_IO_BACKEND_POSIX
from deepspeed.runtime.zero.config import DeepSpeedZeroConfig, DeepSpeedZeroOffloadParamConfig, DeepSpeedZeroOffloadOptimizerConfig
from deepspeed.runtime.zero.offload_config import has_nvme_path, resolve_nvme_path


def test_zero_config_deprecatedfields():
    config = DeepSpeedZeroConfig(**{"cpu_offload_param": True})
    assert isinstance(config.offload_param, DeepSpeedZeroOffloadParamConfig)

    config = DeepSpeedZeroConfig(**{"cpu_offload": True})
    assert isinstance(config.offload_optimizer, DeepSpeedZeroOffloadOptimizerConfig)

    config = DeepSpeedZeroConfig(**{"stage3_gather_fp16_weights_on_model_save": True})
    assert config.gather_16bit_weights_on_model_save == True


def test_zero_config_aliasfields():
    config = DeepSpeedZeroConfig(**{"stage3_prefetch_bucket_size": 12345})
    assert config.prefetch_bucket_size == 12345

    config = DeepSpeedZeroConfig(**{"stage3_param_persistence_threshold": 12345})
    assert config.param_persistence_threshold == 12345

    config = DeepSpeedZeroConfig(**{"stage3_max_reuse_distance": 12345})
    assert config.max_reuse_distance == 12345

    config = DeepSpeedZeroConfig(**{"stage3_gather_16bit_weights_on_model_save": True})
    assert config.gather_16bit_weights_on_model_save == True


def test_zero_config_pipeline_loading_checkpoint():
    for stage in [0, 1, 2]:
        config = DeepSpeedZeroConfig(**{"stage": stage})
        assert config.pipeline_loading_checkpoint == False


def test_zero_config_overlapcomm():
    for stage in [0, 1, 2]:
        config = DeepSpeedZeroConfig(**{"stage": stage})
        assert config.overlap_comm == False

    config = DeepSpeedZeroConfig(**{"stage": 3})
    assert config.overlap_comm == True


def test_zero_config_offload_configs():
    config = DeepSpeedZeroConfig()
    assert config.offload_param is None
    assert config.offload_optimizer is None

    config = DeepSpeedZeroConfig(**{"offload_param": None, "offload_optimizer": None})
    assert config.offload_param is None
    assert config.offload_optimizer is None

    config = DeepSpeedZeroConfig(**{"offload_param": {}, "offload_optimizer": {}})
    assert isinstance(config.offload_param, DeepSpeedZeroOffloadParamConfig)
    assert isinstance(config.offload_optimizer, DeepSpeedZeroOffloadOptimizerConfig)


def test_aio_config_io_backend():
    assert get_aio_config({})[AIO_IO_BACKEND] == AIO_IO_BACKEND_ASYNC
    assert get_aio_config({"aio": {"io_backend": "deepnvme"}})[AIO_IO_BACKEND] == AIO_IO_BACKEND_ASYNC
    assert get_aio_config({"aio": {"io_backend": "simple"}})[AIO_IO_BACKEND] == AIO_IO_BACKEND_POSIX
    assert get_aio_config({"aio": {"io_backend": "posix"}})[AIO_IO_BACKEND] == AIO_IO_BACKEND_POSIX

    with pytest.raises(ValueError, match="Invalid aio.io_backend"):
        get_aio_config({"aio": {"io_backend": "unknown"}})

    with pytest.raises(ValueError, match="requires aio.io_backend='async_io'"):
        get_aio_config({"aio": {"io_backend": "posix", "use_gds": True}})


def test_zero_offload_config_nvme_path_per_local_rank(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")

    config = DeepSpeedZeroConfig(**{
        "offload_param": {
            "device": "nvme",
            "nvme_path_per_local_rank": ["/mnt/raid0/ds_swap", "/mnt/raid1/ds_swap"],
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path_per_local_rank": ["/mnt/raid0/ds_swap", "/mnt/raid1/ds_swap"],
            "buffer_count": 4,
        },
    })

    assert has_nvme_path(config.offload_param) is True
    assert config.offload_param.nvme_path == Path("/mnt/raid1/ds_swap")
    assert config.offload_optimizer.nvme_path == Path("/mnt/raid1/ds_swap")
    assert config.offload_param.nvme_path_per_local_rank == [
        Path("/mnt/raid0/ds_swap"),
        Path("/mnt/raid1/ds_swap"),
    ]

    assert resolve_nvme_path(config.offload_param) == Path("/mnt/raid1/ds_swap")
    assert resolve_nvme_path(config.offload_optimizer) == Path("/mnt/raid1/ds_swap")

    # Parsed configs keep the rank-local nvme_path fixed even if the process
    # environment changes later.
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert resolve_nvme_path(config.offload_param) == Path("/mnt/raid1/ds_swap")

    monkeypatch.setenv("LOCAL_RANK", "2")
    with pytest.raises(ValueError, match="LOCAL_RANK=2"):
        DeepSpeedZeroConfig(**{
            "offload_param": {
                "device": "nvme",
                "nvme_path_per_local_rank": ["/mnt/raid0/ds_swap", "/mnt/raid1/ds_swap"],
            },
        })


def test_superrl_cache_config_uses_single_boolean_control():
    assert get_superrl_cache_config({}).enabled is False
    assert get_superrl_cache_config({"superrl_cache": True}).enabled is True
    assert get_superrl_cache_config({"superrl_cache": False}).enabled is False

    # Existing artifact configs used an object form. Keep it as a compatibility
    # alias, but only the enabled flag controls behavior.
    assert get_superrl_cache_config({"superrl_cache": {"enabled": True, "dram_budget_bytes": 1}}).enabled is True

    with pytest.raises(ValueError, match="superrl_cache must be a boolean"):
        get_superrl_cache_config({"superrl_cache": "true"})


def test_superrl_io_config_uses_single_boolean_control():
    assert get_superrl_io_config({}).enabled is False
    assert get_superrl_io_config({"superrl_io": True}).enabled is True
    assert get_superrl_io_config({"superrl_io": False}).enabled is False

    # Object form remains compatible and can carry read/write GDS tuning.
    config = get_superrl_io_config({"superrl_io": {
        "enabled": True,
        "queue_depth": 32,
        "prefetch_depth": 2,
        "read_thread_count": 8,
        "write_thread_count": 2,
    }})
    assert config.enabled is True
    assert config.prefetch_depth == 2
    assert config.read_thread_count == 8
    assert config.write_thread_count == 2

    with pytest.raises(ValueError, match="read_thread_count"):
        get_superrl_io_config({"superrl_io": {"enabled": True, "read_thread_count": 0}})

    with pytest.raises(ValueError, match="superrl_io must be a boolean"):
        get_superrl_io_config({"superrl_io": "true"})


def test_superrl_sync_config_uses_single_boolean_control():
    assert get_superrl_sync_config({}).enabled is False
    assert get_superrl_sync_config({"superrl_sync": True}).enabled is True
    assert get_superrl_sync_config({"superrl_sync": False}).enabled is False

    # Older artifact-style configs may use an object form. Keep it as a
    # compatibility alias, but the public SuperRL-Sync control is one boolean.
    assert get_superrl_sync_config({"superrl_sync": {"enabled": True, "leaf_policy": "moe"}}).enabled is True

    with pytest.raises(ValueError, match="superrl_sync must be a boolean"):
        get_superrl_sync_config({"superrl_sync": "true"})


def test_zero_offload_optimizer_config_pipeline():
    config = DeepSpeedZeroOffloadOptimizerConfig()
    assert config.pipeline == False

    config = DeepSpeedZeroOffloadOptimizerConfig(**{"pipeline_read": True, "pipeline_write": False})
    assert config.pipeline == True

    config = DeepSpeedZeroOffloadOptimizerConfig(**{"pipeline_read": False, "pipeline_write": True})
    assert config.pipeline == True

    config = DeepSpeedZeroOffloadOptimizerConfig(**{"pipeline_read": True, "pipeline_write": True})
    assert config.pipeline == True


def test_zero_offload_optimizer_config_nvme_pipeline_buffer_count_validation():
    with pytest.raises(ValidationError, match="requires buffer_count >= 12"):
        DeepSpeedZeroOffloadOptimizerConfig(**{"device": "nvme", "pipeline_read": True})

    with pytest.raises(ValidationError, match="requires buffer_count >= 12"):
        DeepSpeedZeroOffloadOptimizerConfig(**{"device": "nvme", "pipeline_write": True})

    with pytest.raises(ValidationError, match="requires buffer_count >= 16"):
        DeepSpeedZeroOffloadOptimizerConfig(**{
            "device": "nvme",
            "pipeline_read": True,
            "pipeline_write": True,
        })

    config = DeepSpeedZeroOffloadOptimizerConfig(**{
        "device": "nvme",
        "pipeline_read": True,
        "buffer_count": 12,
        "buffer_size": 12345,
    })
    assert config.pipeline == True
    assert config.buffer_size == 12345

    config = DeepSpeedZeroOffloadOptimizerConfig(**{
        "device": "nvme",
        "pipeline_write": True,
        "buffer_count": 12,
    })
    assert config.pipeline == True

    config = DeepSpeedZeroOffloadOptimizerConfig(**{
        "device": "nvme",
        "pipeline_read": True,
        "pipeline_write": True,
        "buffer_count": 16,
    })
    assert config.pipeline == True

    with pytest.raises(ValidationError, match="requires buffer_size > 0"):
        DeepSpeedZeroOffloadOptimizerConfig(**{
            "device": "nvme",
            "pipeline_read": True,
            "buffer_count": 12,
            "buffer_size": 0,
        })
