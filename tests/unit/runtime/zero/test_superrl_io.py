# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import pytest

from deepspeed.runtime.swap_tensor.constants import AIO_BLOCK_SIZE, AIO_INTRA_OP_PARALLELISM, AIO_OVERLAP_EVENTS, \
    AIO_QUEUE_DEPTH, AIO_SINGLE_SUBMIT
from deepspeed.runtime.swap_tensor.superrl_io import GDSProbeResult, ensure_superrl_io_gds_ready, \
    make_superrl_io_swap_config, probe_gds_path


def _aio_config():
    return {
        AIO_BLOCK_SIZE: 1024 * 1024,
        AIO_QUEUE_DEPTH: 8,
        AIO_SINGLE_SUBMIT: False,
        AIO_OVERLAP_EVENTS: True,
        AIO_INTRA_OP_PARALLELISM: 1,
    }


def test_superrl_io_disabled_does_not_probe():
    called = False

    def probe_fn(_nvme_path, _aio_config):
        nonlocal called
        called = True
        return GDSProbeResult(ok=False, message="should not be called")

    ensure_superrl_io_gds_ready(SimpleNamespace(enabled=False), None, _aio_config(), probe_fn=probe_fn)

    assert called is False


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
        return GDSProbeResult(ok=False, message="NVMe: Unsupported")

    with pytest.raises(RuntimeError, match="real GPUDirect Storage is not usable"):
        ensure_superrl_io_gds_ready(SimpleNamespace(enabled=True),
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


def test_superrl_io_swap_config_enables_hidden_pipeline_defaults():
    config = make_superrl_io_swap_config(
        SimpleNamespace(buffer_count=4, buffer_size=12345, pipeline=False, pipeline_read=False, pipeline_write=False))

    assert config.pipeline is True
    assert config.pipeline_read is True
    assert config.pipeline_write is True
    assert config.buffer_count == 16
    assert config.buffer_size == 12345


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
