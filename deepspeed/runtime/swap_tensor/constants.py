# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
AIO
"""
AIO_FORMAT = '''
"aio": {
  "io_backend": "async_io",
  "block_size": 1048576,
  "queue_depth": 8,
  "thread_count": 1,
  "intra_op_parallelism": 1,
  "single_submit": false,
  "overlap_events": true,
  "use_gds": false
}
'''
AIO = "aio"
AIO_IO_BACKEND = "io_backend"
AIO_IO_BACKEND_ASYNC = "async_io"
AIO_IO_BACKEND_POSIX = "posix"
AIO_IO_BACKEND_DEFAULT = AIO_IO_BACKEND_ASYNC
AIO_BLOCK_SIZE = "block_size"
AIO_BLOCK_SIZE_DEFAULT = 1048576
AIO_QUEUE_DEPTH = "queue_depth"
AIO_QUEUE_DEPTH_DEFAULT = 8
AIO_INTRA_OP_PARALLELISM = "intra_op_parallelism"
AIO_INTRA_OP_PARALLELISM_DEFAULT = 1
AIO_THREAD_COUNT = "thread_count"
AIO_SINGLE_SUBMIT = "single_submit"
AIO_SINGLE_SUBMIT_DEFAULT = False
AIO_OVERLAP_EVENTS = "overlap_events"
AIO_OVERLAP_EVENTS_DEFAULT = True
AIO_USE_GDS = "use_gds"
AIO_USE_GDS_DEFAULT = False
