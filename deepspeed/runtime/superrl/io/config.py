# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# Config keys + parser for the SuperRL-IO subsystem.
#
# Example ds_config fragment:
#
#   "superrl_io": {
#       "enabled": true,
#       "nvme_devices": ["/mnt/nvme0/ds_swap", "/mnt/nvme1/ds_swap"],
#       "block_size": 16777216,
#       "queue_depth": 32,
#       "intra_op_parallelism": 4,
#       "use_gds": true,
#       "pipelined_adam": true,
#       "double_buffer_bytes": 268435456
#   }

from deepspeed.runtime.config_utils import get_scalar_param, get_list_param

SUPERRL_IO = "superrl_io"

SUPERRL_IO_ENABLED = "enabled"
SUPERRL_IO_ENABLED_DEFAULT = False

SUPERRL_IO_NVME_DEVICES = "nvme_devices"
SUPERRL_IO_NVME_DEVICES_DEFAULT = []

SUPERRL_IO_BLOCK_SIZE = "block_size"
SUPERRL_IO_BLOCK_SIZE_DEFAULT = 16 * 1024 * 1024

SUPERRL_IO_QUEUE_DEPTH = "queue_depth"
SUPERRL_IO_QUEUE_DEPTH_DEFAULT = 32

SUPERRL_IO_INTRA_OP_PARALLELISM = "intra_op_parallelism"
SUPERRL_IO_INTRA_OP_PARALLELISM_DEFAULT = 4

SUPERRL_IO_SINGLE_SUBMIT = "single_submit"
SUPERRL_IO_SINGLE_SUBMIT_DEFAULT = False

SUPERRL_IO_OVERLAP_EVENTS = "overlap_events"
SUPERRL_IO_OVERLAP_EVENTS_DEFAULT = True

SUPERRL_IO_USE_GDS = "use_gds"
SUPERRL_IO_USE_GDS_DEFAULT = False

SUPERRL_IO_PIPELINED_ADAM = "pipelined_adam"
SUPERRL_IO_PIPELINED_ADAM_DEFAULT = False

SUPERRL_IO_DOUBLE_BUFFER_BYTES = "double_buffer_bytes"
SUPERRL_IO_DOUBLE_BUFFER_BYTES_DEFAULT = 256 * 1024 * 1024


class SuperRLIOConfig(object):
    """Parsed view of the `superrl_io` block in `ds_config`.

    Mirrors `DeepSpeedNebulaConfig` in style so the rest of the runtime can
    treat it uniformly.
    """

    def __init__(self, param_dict):
        super().__init__()

        self.enabled = False
        self.nvme_devices = []
        self.block_size = SUPERRL_IO_BLOCK_SIZE_DEFAULT
        self.queue_depth = SUPERRL_IO_QUEUE_DEPTH_DEFAULT
        self.intra_op_parallelism = SUPERRL_IO_INTRA_OP_PARALLELISM_DEFAULT
        self.single_submit = SUPERRL_IO_SINGLE_SUBMIT_DEFAULT
        self.overlap_events = SUPERRL_IO_OVERLAP_EVENTS_DEFAULT
        self.use_gds = SUPERRL_IO_USE_GDS_DEFAULT
        self.pipelined_adam = SUPERRL_IO_PIPELINED_ADAM_DEFAULT
        self.double_buffer_bytes = SUPERRL_IO_DOUBLE_BUFFER_BYTES_DEFAULT

        if param_dict is None or SUPERRL_IO not in param_dict:
            return

        io_dict = param_dict[SUPERRL_IO]
        self._initialize(io_dict)

    def _initialize(self, io_dict):
        self.enabled = get_scalar_param(io_dict, SUPERRL_IO_ENABLED, SUPERRL_IO_ENABLED_DEFAULT)
        self.nvme_devices = get_list_param(io_dict, SUPERRL_IO_NVME_DEVICES, SUPERRL_IO_NVME_DEVICES_DEFAULT)
        self.block_size = get_scalar_param(io_dict, SUPERRL_IO_BLOCK_SIZE, SUPERRL_IO_BLOCK_SIZE_DEFAULT)
        self.queue_depth = get_scalar_param(io_dict, SUPERRL_IO_QUEUE_DEPTH, SUPERRL_IO_QUEUE_DEPTH_DEFAULT)
        self.intra_op_parallelism = get_scalar_param(io_dict, SUPERRL_IO_INTRA_OP_PARALLELISM,
                                                     SUPERRL_IO_INTRA_OP_PARALLELISM_DEFAULT)
        self.single_submit = get_scalar_param(io_dict, SUPERRL_IO_SINGLE_SUBMIT, SUPERRL_IO_SINGLE_SUBMIT_DEFAULT)
        self.overlap_events = get_scalar_param(io_dict, SUPERRL_IO_OVERLAP_EVENTS, SUPERRL_IO_OVERLAP_EVENTS_DEFAULT)
        self.use_gds = get_scalar_param(io_dict, SUPERRL_IO_USE_GDS, SUPERRL_IO_USE_GDS_DEFAULT)
        self.pipelined_adam = get_scalar_param(io_dict, SUPERRL_IO_PIPELINED_ADAM, SUPERRL_IO_PIPELINED_ADAM_DEFAULT)
        self.double_buffer_bytes = get_scalar_param(io_dict, SUPERRL_IO_DOUBLE_BUFFER_BYTES,
                                                    SUPERRL_IO_DOUBLE_BUFFER_BYTES_DEFAULT)

    def stripe_count(self):
        return max(1, len(self.nvme_devices))
