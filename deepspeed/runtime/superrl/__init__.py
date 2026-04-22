# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# SuperRL extensions for DeepSpeed (SC26 artifact).
#
# - `superrl.io`    : coalesced/striped NVMe streaming + pipelined GPU Adam
#                     (paper sec. IV.C).
# - `superrl.cache` : execution-ordered look-ahead parameter cache in host
#                     DRAM (paper sec. IV.B).
