#!/usr/bin/env python3
"""
GH200 NVLink-C2C bandwidth benchmark.

Measures CPU-to-GPU and GPU-to-CPU transfer bandwidth across a range of tensor
sizes, matching the methodology of Figure 7 in the GH200 bandwidth study.

Usage:
    python benchmark_bandwidth.py [--warmup 10] [--iterations 100]
"""

import argparse
import time

import torch

TENSOR_SIZES_MB = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
BYTES_PER_FLOAT32 = 4


def num_elements(size_mb: float) -> int:
    return int(size_mb * 1024 * 1024 / BYTES_PER_FLOAT32)


def measure_bandwidth(
    size_mb: float,
    direction: str,
    warmup: int,
    iterations: int,
) -> float:
    """Return bandwidth in GB/s for a single tensor size and direction."""
    n = num_elements(size_mb)
    nbytes = n * BYTES_PER_FLOAT32

    if direction == "cpu2gpu":
        src = torch.randn(n, dtype=torch.float32, device="cpu", pin_memory=True)
    else:
        src = torch.randn(n, dtype=torch.float32, device="cuda")

    dst_device = "cuda" if direction == "cpu2gpu" else "cpu"

    for _ in range(warmup):
        _ = src.to(dst_device, non_blocking=False)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        _ = src.to(dst_device, non_blocking=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bw_gbs = (nbytes * iterations) / elapsed / 1e9
    return bw_gbs


def run_sweep(warmup: int, iterations: int):
    cpu2gpu_bw = []
    gpu2cpu_bw = []

    for size_mb in TENSOR_SIZES_MB:
        bw = measure_bandwidth(size_mb, "cpu2gpu", warmup, iterations)
        cpu2gpu_bw.append(bw)

    for size_mb in TENSOR_SIZES_MB:
        bw = measure_bandwidth(size_mb, "gpu2cpu", warmup, iterations)
        gpu2cpu_bw.append(bw)

    return cpu2gpu_bw, gpu2cpu_bw


def print_table(cpu2gpu_bw, gpu2cpu_bw):
    header = f"{'Size (MB)':>10} | {'CPU->GPU (GB/s)':>16} | {'GPU->CPU (GB/s)':>16}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for size_mb, c2g, g2c in zip(TENSOR_SIZES_MB, cpu2gpu_bw, gpu2cpu_bw):
        size_str = str(int(size_mb)) if size_mb == int(size_mb) else str(size_mb)
        print(f"{size_str:>10} | {c2g:>16.2f} | {g2c:>16.2f}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="GH200 C2C bandwidth benchmark")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations (default: 3)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of timed iterations (default: 10)")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA device required"
    dev = torch.cuda.get_device_properties(0)
    print(f"Device: {dev.name}  (CUDA {dev.major}.{dev.minor})")
    print(f"Warmup: {args.warmup}  Iterations: {args.iterations}\n")

    cpu2gpu_bw, gpu2cpu_bw = run_sweep(args.warmup, args.iterations)
    print_table(cpu2gpu_bw, gpu2cpu_bw)


if __name__ == "__main__":
    main()
