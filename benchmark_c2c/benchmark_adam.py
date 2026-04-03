"""
Adam latency benchmark for PyTorch Native CPU-Adam and DeepSpeed CPU-Adam.

Reproduces Table 3: Adam latency (s) at 1B, 2B, and 4B parameter counts.

Usage:
    python benchmark_adam.py [--warmup 3] [--iterations 10]
"""

import argparse
import time

import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam

MODEL_SIZES = {
    "1B": 1 * 1024**3 // 4,
    "2B": 2 * 1024**3 // 4,
    "4B": 4 * 1024**3 // 4,
}

NUM_WARMUP = 3
NUM_ITERS = 10


def bench_optimizer(param, optimizer_func, warmup, iterations):
    for p in param:
        p.requires_grad = True
        p.grad = torch.ones_like(p) * 2.0
    optimizer = optimizer_func(param)

    for _ in range(warmup):
        optimizer.step()

    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.step()
    elapsed = time.perf_counter() - start

    return elapsed / iterations


def main():
    parser = argparse.ArgumentParser(description="Adam latency benchmark")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP)
    parser.add_argument("--iterations", type=int, default=NUM_ITERS)
    args = parser.parse_args()

    header = f"{'#Parameter':>12} | {'PT-CPU (s)':>12} | {'CPU-Adam (s)':>14}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for name, numel in MODEL_SIZES.items():
        param_pt = [torch.nn.Parameter(torch.ones(numel, device="cpu"))]
        pt_time = bench_optimizer(param_pt, torch.optim.Adam, args.warmup, args.iterations)
        del param_pt

        param_ds = [torch.nn.Parameter(torch.ones(numel, device="cpu"))]
        ds_time = bench_optimizer(param_ds, DeepSpeedCPUAdam, args.warmup, args.iterations)
        del param_ds

        print(f"{name:>12} | {pt_time:>12.3f} | {ds_time:>14.3f}")

    print(sep)


if __name__ == "__main__":
    main()
