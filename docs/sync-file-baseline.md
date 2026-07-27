# Synchronous File I/O Baseline

This branch is based on DeepSpeed `b35d9eb0` (`v0.18.4`) and intentionally
provides a simple disk-offload baseline.

## Behavior

- ZeRO leaf-module APIs remain available, but the engine clears all leaf tags
  and does not apply config- or threshold-based leaf wrapping.
- Parameter and optimizer swap files are read and written synchronously through
  Python `open`, `read`, and `write` calls.
- DeepNVMe AIO and GDS builders remain in the source tree but are not imported
  or loaded by runtime swapping.
- Fast checkpoint writer requests fall back to `PyFileWriter`.
- `pipeline_read` and `pipeline_write` are always forced to `false`.
- Other DeepSpeed behavior and optimization settings are unchanged.

The compatibility methods retain the existing `async_pread`, `async_pwrite`,
and `wait` interface, but submissions complete inline. This branch is intended
for controlled baseline measurements, not production throughput.

## Limitations

- Dynamic multi-rank MoE execution that requires ZeRO leaf modules is not
  supported.
- Disk I/O stages through CPU tensors; there is no GPU-direct path.
- There is no read/write overlap, prefetch queue, or asynchronous completion.
- Existing `aio` tuning values are accepted for config compatibility but do not
  change the synchronous file backend.
