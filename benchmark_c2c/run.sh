#!/bin/bash
# Bind to NUMA node 0 (CPUs 0-71) and GPU 0 (GPU NUMA node 2 for memory)
NUMA_ARGS="--cpunodebind=0 --membind=0,2"

echo "===== Bandwidth Benchmark (Figure 7) ====="
CUDA_VISIBLE_DEVICES=0 numactl $NUMA_ARGS python benchmark_bandwidth.py "$@"

echo ""
echo "===== Cast vs Move Benchmark (Figure 9) ====="
CUDA_VISIBLE_DEVICES=0 numactl $NUMA_ARGS python benchmark_cast_move.py "$@"

echo ""
echo "===== Adam Latency Benchmark (Table 3) ====="
CUDA_VISIBLE_DEVICES=0 numactl $NUMA_ARGS python benchmark_adam.py "$@"
