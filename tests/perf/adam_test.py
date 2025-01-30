# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
import time

NUM_ITERS = 11


def _test_perf(param, optimizer_func):
    for p in param:
        p.requires_grad = True
        p.grad = torch.ones_like(p) * 2
    optimizer = optimizer_func(param)

    avg = 0
    # import pdb; pdb.set_trace()
    for i in range(NUM_ITERS):
        start = time.time()
        # print(optimizer.state)
        optimizer.step()
        print(optimizer.state)
        stop = time.time()
        if i != 0:
            avg += (stop - start)

    return avg / (NUM_ITERS - 1)


def _main():
    device = 'cpu'
    model_size = 1 * 1024**3
    group_size = [model_size]
    param = [torch.nn.Parameter(torch.ones(size, device=device)) for size in group_size]
    # torch_time = _test_perf(param, torch.optim.Adam)
    torch_time = 0
    ds_time = _test_perf(param, DeepSpeedCPUAdam)
    print(f"Step time: {torch_time=} {ds_time=}")


_main()
