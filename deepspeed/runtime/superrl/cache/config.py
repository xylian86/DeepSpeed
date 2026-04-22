# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

from deepspeed.runtime.config_utils import get_scalar_param

SUPERRL_CACHE = "superrl_cache"
SUPERRL_CACHE_ENABLED = "enabled"
SUPERRL_CACHE_ENABLED_DEFAULT = False
SUPERRL_CACHE_DRAM_BUDGET_BYTES = "dram_budget_bytes"
SUPERRL_CACHE_DRAM_BUDGET_BYTES_DEFAULT = 16 * 1024**3  # 16 GB
SUPERRL_CACHE_WINDOW_SIZE = "window_size"
SUPERRL_CACHE_WINDOW_SIZE_DEFAULT = 0  # 0 = fill to dram_budget_bytes
SUPERRL_CACHE_WARMUP_ITERS = "warmup_iters"
SUPERRL_CACHE_WARMUP_ITERS_DEFAULT = 1
SUPERRL_CACHE_MOE_LEAF_AGGREGATION = "moe_leaf_aggregation"
SUPERRL_CACHE_MOE_LEAF_AGGREGATION_DEFAULT = True


class SuperRLCacheConfig(object):
    """Parsed view of the ``superrl_cache`` block in ``ds_config``.

    Example::

        "superrl_cache": {
            "enabled": true,
            "dram_budget_bytes": 17179869184,
            "warmup_iters": 1,
            "moe_leaf_aggregation": true
        }
    """

    def __init__(self, param_dict):
        self.enabled = False
        self.dram_budget_bytes = SUPERRL_CACHE_DRAM_BUDGET_BYTES_DEFAULT
        self.window_size = SUPERRL_CACHE_WINDOW_SIZE_DEFAULT
        self.warmup_iters = SUPERRL_CACHE_WARMUP_ITERS_DEFAULT
        self.moe_leaf_aggregation = SUPERRL_CACHE_MOE_LEAF_AGGREGATION_DEFAULT

        if param_dict is None or SUPERRL_CACHE not in param_dict:
            return
        d = param_dict[SUPERRL_CACHE]
        self.enabled = get_scalar_param(d, SUPERRL_CACHE_ENABLED, SUPERRL_CACHE_ENABLED_DEFAULT)
        self.dram_budget_bytes = get_scalar_param(d, SUPERRL_CACHE_DRAM_BUDGET_BYTES,
                                                  SUPERRL_CACHE_DRAM_BUDGET_BYTES_DEFAULT)
        self.window_size = get_scalar_param(d, SUPERRL_CACHE_WINDOW_SIZE, SUPERRL_CACHE_WINDOW_SIZE_DEFAULT)
        self.warmup_iters = get_scalar_param(d, SUPERRL_CACHE_WARMUP_ITERS, SUPERRL_CACHE_WARMUP_ITERS_DEFAULT)
        self.moe_leaf_aggregation = get_scalar_param(d, SUPERRL_CACHE_MOE_LEAF_AGGREGATION,
                                                     SUPERRL_CACHE_MOE_LEAF_AGGREGATION_DEFAULT)
