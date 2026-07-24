# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import pytest
import torch

from deepspeed.runtime.swap_tensor import pipelined_optimizer_swapper
from deepspeed.runtime.swap_tensor.constants import AIO_BLOCK_SIZE, AIO_INTRA_OP_PARALLELISM, AIO_OVERLAP_EVENTS, \
    AIO_QUEUE_DEPTH, AIO_SINGLE_SUBMIT, AIO_THREAD_COUNT
from deepspeed.runtime.swap_tensor.superrl_io import GDSProbeResult, ensure_superrl_io_gds_ready, \
    make_superrl_io_swap_config, probe_gds_path
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3


def _aio_config():
    return {
        AIO_BLOCK_SIZE: 1024 * 1024,
        AIO_QUEUE_DEPTH: 8,
        AIO_SINGLE_SUBMIT: False,
        AIO_OVERLAP_EVENTS: True,
        AIO_INTRA_OP_PARALLELISM: 1,
        AIO_THREAD_COUNT: 1,
    }


def test_superrl_io_disabled_does_not_probe():
    called = False

    def probe_fn(_nvme_path, _aio_config):
        nonlocal called
        called = True
        return GDSProbeResult(ok=False, message="should not be called")

    ensure_superrl_io_gds_ready(SimpleNamespace(enabled=False), None, _aio_config(), probe_fn=probe_fn)

    assert called is False


def test_superrl_gds_optimizer_builders_are_imported():
    assert pipelined_optimizer_swapper.GDSBuilder is not None
    assert pipelined_optimizer_swapper.AsyncIOBuilder is not None


def test_superrl_io_requires_nvme_optimizer_offload():
    with pytest.raises(ValueError, match="offload_optimizer"):
        ensure_superrl_io_gds_ready(SimpleNamespace(enabled=True), None, _aio_config(), probe_fn=None)

    with pytest.raises(ValueError, match="device='nvme'"):
        ensure_superrl_io_gds_ready(SimpleNamespace(enabled=True),
                                    SimpleNamespace(device="cpu", nvme_path="/tmp/nvme"),
                                    _aio_config(),
                                    probe_fn=None)

    with pytest.raises(ValueError, match="nvme_path"):
        ensure_superrl_io_gds_ready(SimpleNamespace(enabled=True),
                                    SimpleNamespace(device="nvme", nvme_path=None),
                                    _aio_config(),
                                    probe_fn=None)


def test_superrl_io_fails_when_gds_probe_fails():

    def probe_fn(nvme_path, aio_config):
        assert str(nvme_path) == "/tmp/nvme"
        assert aio_config[AIO_QUEUE_DEPTH] == 8
        assert aio_config[AIO_INTRA_OP_PARALLELISM] == 4
        return GDSProbeResult(ok=False, message="NVMe: Unsupported")

    with pytest.raises(RuntimeError, match="real GPUDirect Storage is not usable"):
        ensure_superrl_io_gds_ready(SimpleNamespace(enabled=True, read_thread_count=4),
                                    SimpleNamespace(device="nvme", nvme_path="/tmp/nvme"),
                                    _aio_config(),
                                    probe_fn=probe_fn)


def test_superrl_io_accepts_successful_gds_probe():

    def probe_fn(_nvme_path, _aio_config):
        return GDSProbeResult(ok=True, message="ok")

    ensure_superrl_io_gds_ready(SimpleNamespace(enabled=True),
                                SimpleNamespace(device="nvme", nvme_path="/tmp/nvme"),
                                _aio_config(),
                                probe_fn=probe_fn)


def test_superrl_io_uses_per_local_rank_nvme_path(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")

    def probe_fn(nvme_path, aio_config):
        assert str(nvme_path) == "/mnt/raid1/ds_swap"
        assert aio_config[AIO_QUEUE_DEPTH] == 8
        return GDSProbeResult(ok=True, message="ok")

    ensure_superrl_io_gds_ready(SimpleNamespace(enabled=True),
                                SimpleNamespace(device="nvme",
                                                nvme_path="/mnt/fallback/ds_swap",
                                                nvme_path_per_local_rank=["/mnt/raid0/ds_swap",
                                                                          "/mnt/raid1/ds_swap"]),
                                _aio_config(),
                                probe_fn=probe_fn)


def test_superrl_io_swap_config_enables_hidden_pipeline_defaults():
    config = make_superrl_io_swap_config(
        SimpleNamespace(buffer_count=4, buffer_size=12345, pipeline=False, pipeline_read=False, pipeline_write=False),
        SimpleNamespace(prefetch_depth=2, read_thread_count=8, write_thread_count=2))

    assert config.pipeline is True
    assert config.pipeline_read is True
    assert config.pipeline_write is True
    assert config.buffer_count == 20
    assert config.buffer_size == 12345
    assert config.gds_prefetch_depth == 2
    assert config.gds_read_intra_op_parallelism == 8
    assert config.gds_write_intra_op_parallelism == 2


def test_superrl_io_swap_config_supports_synchronous_five_buffer_mode():
    config = make_superrl_io_swap_config(
        SimpleNamespace(buffer_count=4, buffer_size=12345, pipeline=False, pipeline_read=False, pipeline_write=False),
        SimpleNamespace(pipeline_read=False,
                        pipeline_write=False,
                        prefetch_depth=4,
                        read_thread_count=2,
                        write_thread_count=2))

    assert config.pipeline is False
    assert config.pipeline_read is False
    assert config.pipeline_write is False
    assert config.buffer_count == 5
    assert config.gds_prefetch_depth == 1


def test_gds_probe_requires_cuda_accelerator(monkeypatch):

    class _FakeAccelerator:

        def device_name(self):
            return "cpu"

        def is_available(self):
            return False

    monkeypatch.setattr("deepspeed.runtime.swap_tensor.superrl_io.get_accelerator", lambda: _FakeAccelerator())

    result = probe_gds_path("/tmp/nvme", _aio_config())

    assert result.ok is False
    assert "CUDA accelerator" in result.message


def test_gds_probe_uses_current_cuda_device(monkeypatch, tmp_path):
    captured = {}

    class _FakeAccelerator:

        def device_name(self):
            return "cuda"

        def is_available(self):
            return True

        def current_device(self):
            return 1

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("deepspeed.runtime.swap_tensor.superrl_io.get_accelerator", lambda: _FakeAccelerator())
    monkeypatch.setattr("deepspeed.runtime.swap_tensor.superrl_io.subprocess.run", fake_run)

    result = probe_gds_path(tmp_path, _aio_config())

    assert result.ok is True
    assert captured["cmd"][-1] == "1"
    assert "torch.cuda.set_device(device_index)" in captured["cmd"][2]


def _stage3_mode(superrl_io_enabled=True,
                 low_host_mem_enabled=True,
                 disable_superrl_io=True,
                 pipeline_read=True,
                 pipeline_write=True):
    optimizer = DeepSpeedZeroOptimizer_Stage3.__new__(DeepSpeedZeroOptimizer_Stage3)
    optimizer.superrl_io_enabled = superrl_io_enabled
    optimizer.superrl_low_host_mem_enabled = low_host_mem_enabled
    optimizer.superrl_low_host_mem_config = SimpleNamespace(disable_superrl_io=disable_superrl_io)
    optimizer.superrl_io_config = SimpleNamespace(pipeline_read=pipeline_read, pipeline_write=pipeline_write)
    optimizer.offload_optimizer = True
    return optimizer


def test_low_host_mem_disables_superrl_io_runtime():
    optimizer = _stage3_mode()

    assert optimizer._resolve_superrl_io_active() is False


def test_low_host_mem_rejects_unsupported_superrl_io_override():
    optimizer = _stage3_mode(disable_superrl_io=False)

    with pytest.raises(ValueError, match="supports SuperRL-IO only"):
        optimizer._resolve_superrl_io_active()


def test_low_host_mem_allows_synchronous_superrl_io():
    optimizer = _stage3_mode(disable_superrl_io=False, pipeline_read=False, pipeline_write=False)

    assert optimizer._resolve_superrl_io_active() is True


def test_low_host_mem_optimizer_step_uses_base_optimizer():
    class _FakeOptimizer:

        def __init__(self):
            self.param_groups = [{"params": []}]
            self.step_count = 0

        def step(self):
            self.step_count += 1

    optimizer = _stage3_mode()
    optimizer.superrl_io_active = optimizer._resolve_superrl_io_active()
    optimizer.superrl_io_optimizer = _FakeOptimizer()
    optimizer.optimizer = _FakeOptimizer()
    optimizer.fp32_partitioned_groups_flat = [torch.nn.Parameter(torch.ones(1))]
    optimizer.sub_group_to_group_id = {0: 0}
    optimizer.subgroup_to_device = {0: "cpu"}
    optimizer._swappable_optimizer_subgroup = lambda _sub_group_id: True
    optimizer.torch_autocast_gradscaler = None
    optimizer.zenflow = False

    optimizer._optimizer_step(0)

    assert optimizer.optimizer.step_count == 1
    assert optimizer.superrl_io_optimizer.step_count == 0
