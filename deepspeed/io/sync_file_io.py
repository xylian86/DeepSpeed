# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""Blocking tensor file I/O used by the synchronous baseline."""

import os

import torch


class SyncFileIOHandle:
    """DeepNVMe-compatible handle backed by Python ``read`` and ``write``.

    The async methods intentionally execute inline. ``wait`` only reports and
    clears the number of operations completed since the previous call.
    """

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self._completed = 0

    @staticmethod
    def _byte_view(tensor):
        tensor = tensor.detach()
        if not tensor.is_contiguous():
            raise ValueError("SyncFileIOHandle requires contiguous tensors")
        return tensor.view(torch.uint8)

    @staticmethod
    def _write(buffer, filename, file_offset):
        if not isinstance(filename, (str, bytes, os.PathLike)):
            raise TypeError("SyncFileIOHandle requires a file path")

        byte_view = SyncFileIOHandle._byte_view(buffer)
        cpu_view = byte_view if byte_view.device.type == "cpu" else byte_view.cpu()
        payload = memoryview(cpu_view.numpy())

        path = os.fspath(filename)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        mode = "r+b" if os.path.exists(path) else "w+b"
        with open(path, mode) as file:
            file.seek(file_offset)
            written = file.write(payload)
            if written != byte_view.numel():
                raise OSError(f"Short write to {path}: expected {byte_view.numel()} bytes, wrote {written}")

    @staticmethod
    def _read(buffer, filename, file_offset):
        if not isinstance(filename, (str, bytes, os.PathLike)):
            raise TypeError("SyncFileIOHandle requires a file path")

        byte_view = SyncFileIOHandle._byte_view(buffer)
        path = os.fspath(filename)
        with open(path, "rb") as file:
            file.seek(file_offset)
            payload = file.read(byte_view.numel())
        if len(payload) != byte_view.numel():
            raise OSError(f"Short read from {path}: expected {byte_view.numel()} bytes, read {len(payload)}")

        source = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        if byte_view.device.type != "cpu":
            source = source.to(byte_view.device)
        byte_view.copy_(source)

    def async_pwrite(self, buffer, filename, file_offset=0):
        self._write(buffer, filename, file_offset)
        self._completed += 1
        return 0

    def async_pread(self, buffer, filename, file_offset=0):
        self._read(buffer, filename, file_offset)
        self._completed += 1
        return 0

    def pwrite(self, buffer, filename, validate=False, async_op=False):
        del validate
        if async_op:
            return self.async_pwrite(buffer, filename, 0)
        self._write(buffer, filename, 0)
        return 1

    def pread(self, buffer, filename, validate=False, async_op=False):
        del validate
        if async_op:
            return self.async_pread(buffer, filename, 0)
        self._read(buffer, filename, 0)
        return 1

    def wait(self):
        completed = self._completed
        self._completed = 0
        return completed

    @staticmethod
    def new_cpu_locked_tensor(numel, example_tensor):
        return torch.empty(numel, dtype=example_tensor.dtype, device="cpu").pin_memory()

    @staticmethod
    def free_cpu_locked_tensor(tensor):
        del tensor
        return True

    @staticmethod
    def pin_device_tensor(tensor):
        del tensor
        return True

    @staticmethod
    def unpin_device_tensor(tensor):
        del tensor
        return True
