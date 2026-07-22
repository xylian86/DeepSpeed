# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import get_scalar_param
from deepspeed.runtime.swap_tensor.constants import *
from deepspeed.accelerator import get_accelerator

AIO_DEFAULT_DICT = {
    AIO_IO_BACKEND: AIO_IO_BACKEND_DEFAULT,
    AIO_BLOCK_SIZE: AIO_BLOCK_SIZE_DEFAULT,
    AIO_QUEUE_DEPTH: AIO_QUEUE_DEPTH_DEFAULT,
    AIO_INTRA_OP_PARALLELISM: AIO_INTRA_OP_PARALLELISM_DEFAULT,
    AIO_THREAD_COUNT: AIO_INTRA_OP_PARALLELISM_DEFAULT,
    AIO_SINGLE_SUBMIT: AIO_SINGLE_SUBMIT_DEFAULT,
    AIO_OVERLAP_EVENTS: AIO_OVERLAP_EVENTS_DEFAULT,
    AIO_USE_GDS: AIO_USE_GDS_DEFAULT
}


def _get_aio_backend(aio_dict):
    backend = get_scalar_param(aio_dict, AIO_IO_BACKEND, AIO_IO_BACKEND_DEFAULT)
    backend = backend.lower() if isinstance(backend, str) else backend
    aliases = {
        AIO_IO_BACKEND_ASYNC: AIO_IO_BACKEND_ASYNC,
        "aio": AIO_IO_BACKEND_ASYNC,
        "async": AIO_IO_BACKEND_ASYNC,
        "deepnvme": AIO_IO_BACKEND_ASYNC,
        AIO_IO_BACKEND_POSIX: AIO_IO_BACKEND_POSIX,
        "sync": AIO_IO_BACKEND_POSIX,
        "simple": AIO_IO_BACKEND_POSIX,
    }
    if backend not in aliases:
        raise ValueError(
            f"Invalid aio.{AIO_IO_BACKEND}: {backend}. Supported values: "
            f"{AIO_IO_BACKEND_ASYNC}, {AIO_IO_BACKEND_POSIX}.")
    return aliases[backend]


def _get_aio_parallelism(aio_dict):
    """Accept DeepSpeed's legacy thread_count as an alias for intra_op_parallelism."""
    thread_count = get_scalar_param(aio_dict, AIO_THREAD_COUNT, AIO_INTRA_OP_PARALLELISM_DEFAULT)
    parallelism = get_scalar_param(aio_dict, AIO_INTRA_OP_PARALLELISM, thread_count)
    return max(1, int(parallelism))


def get_aio_config(param_dict):
    if AIO in param_dict.keys() and param_dict[AIO] is not None:
        aio_dict = param_dict[AIO]
        intra_op_parallelism = _get_aio_parallelism(aio_dict)
        aio_config = {
            AIO_IO_BACKEND:
            _get_aio_backend(aio_dict),
            AIO_BLOCK_SIZE:
            get_scalar_param(aio_dict, AIO_BLOCK_SIZE, AIO_BLOCK_SIZE_DEFAULT),
            AIO_QUEUE_DEPTH:
            get_scalar_param(aio_dict, AIO_QUEUE_DEPTH, AIO_QUEUE_DEPTH_DEFAULT),
            AIO_INTRA_OP_PARALLELISM:
            intra_op_parallelism,
            AIO_THREAD_COUNT:
            intra_op_parallelism,
            AIO_SINGLE_SUBMIT:
            get_scalar_param(aio_dict, AIO_SINGLE_SUBMIT, AIO_SINGLE_SUBMIT_DEFAULT),
            AIO_OVERLAP_EVENTS:
            get_scalar_param(aio_dict, AIO_OVERLAP_EVENTS, AIO_OVERLAP_EVENTS_DEFAULT),
            AIO_USE_GDS:
            get_scalar_param(aio_dict, AIO_USE_GDS, AIO_USE_GDS_DEFAULT)
        }

        if aio_config[AIO_USE_GDS]:
            assert get_accelerator().device_name() == 'cuda', 'GDS currently only supported for CUDA accelerator'
            if aio_config[AIO_IO_BACKEND] == AIO_IO_BACKEND_POSIX:
                raise ValueError("aio.use_gds=true requires aio.io_backend='async_io'.")

        return aio_config

    return AIO_DEFAULT_DICT
