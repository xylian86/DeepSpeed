# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.io import MockFileWriter, PyFileWriter
from .constants import *


class CheckpointWriterFactory(object):

    def __init__(self, writer_config, aio_config, dp_writer_config):
        self._type = writer_config[CHECKPOINT_WRITER_TYPE]
        self._io_buffer_size = writer_config[CHECKPOINT_IO_BUFFER_SIZE]
        self._io_buffer_double = writer_config[CHECKPOINT_IO_BUFFER_DOUBLE]
        self._data_parallel_writer = dp_writer_config
        self._io_multiplier = writer_config[CHECKPOINT_IO_MULTIPLIER]
        if self._data_parallel_writer.pure_dp:
            self._show_statistics = writer_config[CHECKPOINT_IO_STATISTICS] and self._data_parallel_writer is not None
        else:
            self._show_statistics = writer_config[CHECKPOINT_IO_STATISTICS] and self._data_parallel_writer is not None
        self._io_buffer = None
        self._dnvme_handle = None
        self._writer = None
        self._use_gds = False
        self._fast_fallback = self._type == CheckpointWriterType.FAST

        if self._fast_fallback:
            self._type = CheckpointWriterType.PYTHON
        print(
            f'WriterFactory: self._data_parallel_writer={self._data_parallel_writer} self._show_statistics={self._show_statistics}'
        )

    def create_writer(self, file_path, optimize_dp_state):
        assert self._writer is None, \
            f'Cannot create checkpoint writer for {file_path} because writer is currently used for {self._writer.file_path()}.\
            Must call writer.release() before reusing to avoid this error.'

        if self._type == CheckpointWriterType.MOCK:
            self._writer = MockFileWriter(file_path)
        elif self._type == CheckpointWriterType.PYTHON:
            if self._fast_fallback and optimize_dp_state:
                num_parallel_writers = self._data_parallel_writer.world_size * self._io_multiplier
                writer_rank = self._data_parallel_writer.rank
                file_path = f'{file_path}-{writer_rank}.{num_parallel_writers}'
            self._writer = PyFileWriter(file_path)
        else:
            raise ValueError(f"Unsupported checkpoint writer type: {self._type}")

        return self._writer

    def release_writer(self):
        self._writer.close()
        if self._show_statistics:
            self._writer._dump_state()
        self._writer = None
