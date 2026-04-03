#!/usr/bin/env python3
"""
GH200 cast-vs-move benchmark (GPU -> CPU direction).

Compares two strategies for getting fp32 data from GPU fp16 tensors to CPU:

  Cast_cpu <- Move_fp16:
    GPU fp16 -> move fp16 to CPU -> cast to fp32 on CPU

  Cast_gpu <- Move_fp32:
    GPU fp16 -> cast to fp32 on GPU -> move fp32 to CPU

Reproduces Figure 9: time cost comparison for casting on GPU vs CPU
(including data transfer overhead).

Usage:
    python benchmark_cast_move.py [--warmup 10] [--iterations 100]
"""

import argparse
import time

import torch

TENSOR_SIZES_MB = [16, 32, 64, 128, 256, 512, 1024, 2048]
BYTES_PER_FP16 = 2


def num_elements(size_mb: int) -> int:
    return size_mb * 1024 * 1024 // BYTES_PER_FP16


def measure_cast_cpu(n: int, warmup: int, iterations: int) -> float:
    """GPU fp16 -> move fp16 to CPU (pinned) -> cast fp32 on CPU. Return ms."""
    src = torch.randn(n, dtype=torch.float16, device="cuda")
    dst_fp16 = torch.empty(n, dtype=torch.float16, device="cpu", pin_memory=True)
    dst_fp32 = torch.empty(n, dtype=torch.float32, device="cpu", pin_memory=True)

    for _ in range(warmup):
        dst_fp16.copy_(src)
        dst_fp32.copy_(dst_fp16)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        dst_fp16.copy_(src)
        dst_fp32.copy_(dst_fp16)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000.0


def measure_cast_gpu(n: int, warmup: int, iterations: int) -> float:
    """GPU fp16 -> cast fp32 on GPU -> move fp32 to CPU (pinned). Return ms."""
    src = torch.randn(n, dtype=torch.float16, device="cuda")
    dst_fp32 = torch.empty(n, dtype=torch.float32, device="cpu", pin_memory=True)

    for _ in range(warmup):
        tmp = src.float()
        dst_fp32.copy_(tmp)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        tmp = src.float()
        dst_fp32.copy_(tmp)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000.0


def run_sweep(warmup: int, iterations: int):
    cast_cpu_times = []
    cast_gpu_times = []

    for size_mb in TENSOR_SIZES_MB:
        n = num_elements(size_mb)
        t = measure_cast_cpu(n, warmup, iterations)
        cast_cpu_times.append(t)

    for size_mb in TENSOR_SIZES_MB:
        n = num_elements(size_mb)
        t = measure_cast_gpu(n, warmup, iterations)
        cast_gpu_times.append(t)

    return cast_cpu_times, cast_gpu_times


def print_table(cast_cpu_times, cast_gpu_times):
    header = (
        f"{'Size (MB)':>10} | "
        f"{'Cast_cpu, Move_fp16 (ms)':>25} | "
        f"{'Cast_gpu, Move_fp32 (ms)':>25}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for size_mb, t_cpu, t_gpu in zip(TENSOR_SIZES_MB, cast_cpu_times, cast_gpu_times):
        print(f"{size_mb:>10} | {t_cpu:>25.2f} | {t_gpu:>25.2f}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="GH200 cast-vs-move time cost benchmark")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations (default: 3)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of timed iterations (default: 10)")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA device required"
    dev = torch.cuda.get_device_properties(0)
    print(f"Device: {dev.name}  (CUDA {dev.major}.{dev.minor})")
    print(f"Warmup: {args.warmup}  Iterations: {args.iterations}\n")

    cast_cpu_times, cast_gpu_times = run_sweep(args.warmup, args.iterations)
    print_table(cast_cpu_times, cast_gpu_times)


if __name__ == "__main__":
    main()
