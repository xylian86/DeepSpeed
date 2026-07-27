# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from unit.common import DistributedTest

import deepspeed
from deepspeed.io import PyFileWriter, SyncFileIOHandle
from deepspeed.runtime.engine import _clear_zero_leaf_module_flags
from deepspeed.runtime.model_checkpointing.constants import (
    CHECKPOINT_IO_BUFFER_DOUBLE,
    CHECKPOINT_IO_BUFFER_SIZE,
    CHECKPOINT_IO_MULTIPLIER,
    CHECKPOINT_IO_STATISTICS,
    CHECKPOINT_WRITER_TYPE,
    CheckpointWriterType,
)
from deepspeed.runtime.model_checkpointing.writer_factory import CheckpointWriterFactory
from deepspeed.runtime.zero.offload_config import DeepSpeedZeroOffloadOptimizerConfig
from deepspeed.utils import set_z3_leaf_modules, z3_leaf_module


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sync_file_io_round_trip(tmp_path, dtype):
    source = torch.arange(257, dtype=torch.float32).to(dtype)
    destination = torch.empty_like(source)
    path = tmp_path / f"tensor-{dtype}.bin"
    handle = SyncFileIOHandle()

    assert handle.async_pwrite(source, path, 0) == 0
    assert handle.wait() == 1
    assert handle.async_pread(destination, path, 0) == 0
    assert handle.wait() == 1
    assert torch.equal(destination, source)
    assert path.stat().st_size == source.numel() * source.element_size()


def test_sync_file_io_wait_counts_completed_operations(tmp_path):
    handle = SyncFileIOHandle()
    tensors = [torch.tensor([value], dtype=torch.int64) for value in range(3)]

    for index, tensor in enumerate(tensors):
        handle.async_pwrite(tensor, tmp_path / f"{index}.bin", 0)

    assert handle.wait() == len(tensors)
    assert handle.wait() == 0


def test_sync_file_io_rejects_short_reads(tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(b"short")

    with pytest.raises(OSError, match="Short read"):
        SyncFileIOHandle().async_pread(torch.empty(8, dtype=torch.uint8), path, 0)


def test_fast_checkpoint_writer_falls_back_to_python(tmp_path):
    writer_config = {
        CHECKPOINT_WRITER_TYPE: CheckpointWriterType.FAST,
        CHECKPOINT_IO_BUFFER_SIZE: 1024,
        CHECKPOINT_IO_BUFFER_DOUBLE: False,
        CHECKPOINT_IO_MULTIPLIER: 1,
        CHECKPOINT_IO_STATISTICS: False,
    }
    dp_config = SimpleNamespace(pure_dp=True, world_size=1, rank=0, global_rank=0)
    factory = CheckpointWriterFactory(writer_config, aio_config={}, dp_writer_config=dp_config)

    writer = factory.create_writer(str(tmp_path / "checkpoint.bin"), optimize_dp_state=False)
    assert isinstance(writer, PyFileWriter)
    writer.write(b"checkpoint")
    factory.release_writer()
    assert (tmp_path / "checkpoint.bin").read_bytes() == b"checkpoint"

    factory = CheckpointWriterFactory(writer_config, aio_config={}, dp_writer_config=dp_config)
    writer = factory.create_writer(str(tmp_path / "sharded.bin"), optimize_dp_state=True)
    writer.write(b"shard")
    factory.release_writer()
    assert (tmp_path / "sharded.bin-0.1").read_bytes() == b"shard"


def test_pipeline_flags_are_forced_off():
    config = DeepSpeedZeroOffloadOptimizerConfig(pipeline_read=True, pipeline_write=True)

    assert config.pipeline_read is False
    assert config.pipeline_write is False
    assert config.pipeline is False


def test_preconfigured_leaf_tags_are_cleared():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    set_z3_leaf_modules(model, [torch.nn.Linear])
    assert z3_leaf_module(model[0])

    assert _clear_zero_leaf_module_flags(model) == 1
    assert not z3_leaf_module(model[0])


def test_runtime_swapper_modules_do_not_expose_deepnvme_builders():
    from deepspeed.runtime.model_checkpointing import writer_factory
    from deepspeed.runtime.swap_tensor import (
        partitioned_optimizer_swapper,
        partitioned_param_swapper,
        pipelined_optimizer_swapper,
    )

    for module in (
        writer_factory,
        partitioned_optimizer_swapper,
        partitioned_param_swapper,
        pipelined_optimizer_swapper,
    ):
        assert not hasattr(module, "AsyncIOBuilder")
        assert not hasattr(module, "GDSBuilder")


class TestEngineDisablesLeafModules(DistributedTest):
    world_size = 1

    def test_engine_clears_explicit_leaf_tags(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4))
        set_z3_leaf_modules(model, [torch.nn.Linear])
        assert any(z3_leaf_module(module) for module in model.modules())

        config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "zero_optimization": {
                "stage": 3,
                "stage3_param_persistence_threshold": 0,
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)

        assert not any(z3_leaf_module(module) for module in model.modules())
        assert len(engine.optimizer.leaf_parameters) == 0
        engine.destroy()
