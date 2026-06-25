# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping tensors to/from (NVMe) storage devices.
"""

import time

import torch
from deepspeed.utils.logging import logger
from deepspeed.accelerator import get_accelerator

from deepspeed import comm as dist

MIN_AIO_BYTES = 1024**2
AIO_ALIGNED_BYTES = 1024
MIN_SWAPPABLE_BYTES = MIN_AIO_BYTES


def _buffer_nbytes(buffer):
    nbytes = buffer.nbytes
    return nbytes() if callable(nbytes) else nbytes


def is_direct_io_buffer(buffer):
    return buffer.is_cuda or get_accelerator().is_pinned(buffer)


def _same_io_buffer_kind(left, right):
    return left.device == right.device and left.dtype == right.dtype and left.layout == right.layout


def _same_storage(left, right):
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _can_coalesce_io_buffer(previous, current_buffer, current_path, expected_file_offset, current_file_offset):
    if previous.path != current_path:
        return False
    if expected_file_offset != current_file_offset:
        return False
    if not _same_io_buffer_kind(previous.buffer, current_buffer):
        return False
    if not previous.buffer.is_contiguous() or not current_buffer.is_contiguous():
        return False
    if not _same_storage(previous.base_buffer, current_buffer):
        return False
    return previous.next_data_ptr == current_buffer.data_ptr()


class _CoalescedIOBuffer(object):

    def __init__(self, buffer, path, offset):
        self.base_buffer = buffer
        self.buffer = buffer
        self.path = path
        self.offset = offset
        self.numel = buffer.numel()
        self.nbytes = _buffer_nbytes(buffer)
        self.next_data_ptr = buffer.data_ptr() + self.nbytes

    def append(self, buffer):
        new_numel = self.numel + buffer.numel()
        storage_nbytes = self.base_buffer.untyped_storage().nbytes()
        storage_numel = storage_nbytes // self.base_buffer.element_size()
        if self.base_buffer.storage_offset() + new_numel > storage_numel:
            raise RuntimeError("Coalesced I/O buffer exceeds base storage")

        buffer_nbytes = _buffer_nbytes(buffer)
        self.numel = new_numel
        self.nbytes += buffer_nbytes
        self.next_data_ptr += buffer_nbytes
        self.buffer = torch.as_strided(self.base_buffer, (self.numel, ), (1, ))


def _coalesce_io_buffers(tensor_buffers, swap_paths, swap_offsets):
    coalesced = []
    for buffer, path, offset in zip(tensor_buffers, swap_paths, swap_offsets):
        if coalesced and _can_coalesce_io_buffer(coalesced[-1], buffer, path,
                                                 coalesced[-1].offset + coalesced[-1].nbytes, offset):
            try:
                coalesced[-1].append(buffer)
                continue
            except RuntimeError:
                pass
        coalesced.append(_CoalescedIOBuffer(buffer, path, offset))

    return coalesced


def swap_in_tensors(swap_handle, tensor_buffers, swap_paths, swap_offsets=None):
    if swap_offsets is None:
        swap_offsets = [0] * len(tensor_buffers)
    assert len(tensor_buffers) == len(swap_paths)
    assert len(tensor_buffers) == len(swap_offsets)
    requests = _coalesce_io_buffers(tensor_buffers, swap_paths, swap_offsets)
    for request in requests:
        assert (swap_handle.async_pread(request.buffer, request.path, request.offset) == 0)
    return len(requests)


def swap_out_tensors(swap_handle, tensor_buffers, swap_paths, swap_offsets=None):
    if swap_offsets is None:
        swap_offsets = [0] * len(tensor_buffers)
    assert len(tensor_buffers) == len(swap_paths)
    assert len(tensor_buffers) == len(swap_offsets)
    requests = _coalesce_io_buffers(tensor_buffers, swap_paths, swap_offsets)
    for request in requests:
        assert (swap_handle.async_pwrite(request.buffer, request.path, request.offset) == 0)
    return len(requests)


def print_object(obj, name, exclude_list=[]):
    logger.info('{}:'.format(name))
    for arg in sorted(vars(obj)):
        if arg not in exclude_list:
            dots = '.' * (29 - len(arg))
            logger.info('  {} {} {}'.format(arg, dots, getattr(obj, arg)))


def print_rank_0(message, debug=False, force=False):
    if dist.get_rank() == 0 and (debug or force):
        print(message)


class SwapBuffer(object):

    def __init__(self, buffer):
        self.buffer = buffer
        self.reset()

    def reset(self):
        self.offset = 0
        self.swap_tensors = {}
        self.compute_tensors = {}
        self.swap_paths = {}
        self.swap_offsets = {}
        self.num_elem = 0

    def insert_tensor(self, tensor, swap_path, aligned_numel, swap_offset=0):
        swap_tensor, compute_tensor = self.allocate_tensor(swap_path, tensor.numel(), aligned_numel, swap_offset)
        compute_tensor.data.copy_(tensor.data)
        return swap_tensor, compute_tensor

    def allocate_tensor(self, swap_path, numel, aligned_numel, swap_offset=0):
        assert self.has_space(aligned_numel)
        assert self.offset not in self.swap_tensors

        allocate_offset = self.offset
        swap_tensor = self.buffer.narrow(0, allocate_offset, aligned_numel)
        dest_tensor = swap_tensor.narrow(0, 0, numel)

        self.swap_tensors[allocate_offset] = swap_tensor
        self.compute_tensors[allocate_offset] = dest_tensor
        self.swap_paths[allocate_offset] = swap_path
        self.swap_offsets[allocate_offset] = swap_offset
        self.offset += aligned_numel
        self.num_elem += numel

        return self.swap_tensors[allocate_offset], self.compute_tensors[allocate_offset]

    def has_space(self, numel):
        return (self.offset + numel) <= self.buffer.numel()

    def get_swap_tensors(self):
        return [tensor for tensor in self.swap_tensors.values()]

    def get_swap_paths(self):
        return [path for path in self.swap_paths.values()]

    def get_swap_offsets(self):
        return [offset for offset in self.swap_offsets.values()]

    def get_compute_tensors(self):
        return [tensor for tensor in self.compute_tensors.values()]

    def get_num_elem(self):
        return self.num_elem

    def get_swap_tensor(self, offset):
        return self.swap_tensors.get(offset, None)

    def get_compute_tensor(self, offset):
        return self.compute_tensors.get(offset, None)

    def get_swap_path(self, offset):
        return self.swap_paths(offset, None)


class SwapBufferPool(object):

    def __init__(self, buffers):
        assert all([is_direct_io_buffer(buf) for buf in buffers])
        self.buffers = [SwapBuffer(buf) for buf in buffers]
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        for buffer in self.buffers:
            buffer.reset()

    def allocate_tensor(self, numel, swap_path, aligned_numel, swap_offset=0):
        if self.has_space(aligned_numel):
            swap_tensor, compute_tensor = self._get_current_buffer().allocate_tensor(swap_path, numel, aligned_numel,
                                                                                    swap_offset)
            return swap_tensor, compute_tensor

        return None, None

    def insert_tensor(self, tensor, swap_path, aligned_numel, swap_offset=0):
        if self.has_space(aligned_numel):
            swap_tensor, compute_tensor = self._get_current_buffer().insert_tensor(tensor, swap_path, aligned_numel,
                                                                                  swap_offset)
            return swap_tensor, compute_tensor

        return None, None

    def get_swap_tensors(self):
        swap_tensors = []
        for buffer in self._get_used_buffers():
            swap_tensors += buffer.get_swap_tensors()

        return swap_tensors

    def get_swap_paths(self):
        swap_paths = []
        for buffer in self._get_used_buffers():
            swap_paths += buffer.get_swap_paths()

        return swap_paths

    def get_swap_offsets(self):
        swap_offsets = []
        for buffer in self._get_used_buffers():
            swap_offsets += buffer.get_swap_offsets()

        return swap_offsets

    def get_compute_tensors(self):
        compute_tensors = []
        for buffer in self._get_used_buffers():
            compute_tensors += buffer.get_compute_tensors()

        return compute_tensors

    def has_space(self, numel):
        if self._get_current_buffer().has_space(numel):
            return True

        if self.current_index == len(self.buffers) - 1:
            return False

        self.current_index += 1
        return self._get_current_buffer().has_space(numel)

    def swap_out(self, aio_handle, async_op=False):
        swap_tensors = self.get_swap_tensors()
        swap_paths = self.get_swap_paths()
        swap_offsets = self.get_swap_offsets()
        assert all([p is not None for p in swap_paths])

        num_swap_ops = swap_out_tensors(aio_handle, swap_tensors, swap_paths, swap_offsets)

        if not async_op:
            assert num_swap_ops == aio_handle.wait()
        return num_swap_ops

    def swap_in(self, aio_handle, async_op=False):
        swap_tensors = self.get_swap_tensors()
        swap_paths = self.get_swap_paths()
        swap_offsets = self.get_swap_offsets()
        assert all([p is not None for p in swap_paths])

        num_swap_ops = swap_in_tensors(aio_handle, swap_tensors, swap_paths, swap_offsets)

        if not async_op:
            assert num_swap_ops == aio_handle.wait()
        return num_swap_ops

    def _get_current_buffer(self):
        return self.buffers[self.current_index]

    def _get_used_buffers(self):
        return self.buffers[:self.current_index + 1]


class SwapBufferLease(object):

    def __init__(self, manager, buffers, owner):
        self.manager = manager
        self.buffers = buffers
        self.owner = owner
        self.released = False

    def release(self):
        if self.released:
            raise RuntimeError(f"Swap buffer lease for {self.owner} was released more than once")
        self.manager.free(self.buffers)
        self.released = True

    def __len__(self):
        return len(self.buffers)

    def __iter__(self):
        return iter(self.buffers)

    def __getitem__(self, index):
        return self.buffers[index]


class SwapBufferManager(object):

    def __init__(self,
                 num_elems,
                 count,
                 dtype,
                 name='swap_buffer',
                 lazy=False,
                 device='cpu',
                 pin_memory_fn=None,
                 unpin_memory_fn=None):
        self.name = name
        self.lazy = lazy
        self.num_elems = num_elems
        self.count = count
        self.dtype = dtype
        self.device = device
        self.pin_memory_fn = pin_memory_fn
        self.unpin_memory_fn = unpin_memory_fn
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        self.pin_memory_time_sec = 0
        self.all_buffers = [None for _ in range(count)]
        self.buffer_numel = [0 for _ in range(count)]
        if not lazy:
            for index in range(count):
                self._allocate_slot(index=index, num_elems=num_elems)
        self.free_buffer_index = [i for i in range(count)]
        self.used_buffer_index = {}
        self.used_buffer_numel = {}
        self.buffer_bytes = self.element_size * num_elems
        self.capacity_bytes = self.element_size * num_elems * count
        self.total_bytes = self._pinned_bytes()
        self.gigabytes = self.total_bytes / (1024**3)
        self.max_pinned_bytes = self.total_bytes
        self.num_allocations = 0
        self.num_failed_allocations = 0
        self.num_buffer_allocations = count if not lazy else 0
        self.num_buffer_reallocations = 0
        self.max_allocated_buffers = 0
        self.max_allocated_bytes = 0
        self.max_requested_in_use_bytes = 0
        self.max_requested_num_elems = 0
        self.max_requested_count = 0
        self.max_requested_bytes = 0

        if dist.get_rank() == 0:
            exclude_list = ['all_buffers']
            print_object(obj=self, name='SwapBufferManager', exclude_list=exclude_list)
            summary = (
                f"SwapBufferManager[{self.name}] initialized {count} {'lazy ' if lazy else ''}buffer slot(s), "
                f"buffer_num_elems={self.num_elems}, "
                f"max_buffer_size={self.buffer_bytes / (1024**3):.2f} GiB, "
                f"pinned_pool_size={self.total_bytes / (1024**3):.2f} GiB, "
                f"max_pool_capacity={self.capacity_bytes / (1024**3):.2f} GiB, "
                f"pin_memory_time={self.pin_memory_time_sec:.3f} sec")
            logger.info(summary)
            print_rank_0(summary, force=True)

    def _allocate_slot(self, index, num_elems):
        start_time = time.time()
        if self.all_buffers[index] is not None and self.unpin_memory_fn is not None:
            self.unpin_memory_fn(self.all_buffers[index])

        buffer = torch.empty(num_elems, device=self.device, dtype=self.dtype)
        if self.pin_memory_fn is not None:
            buffer = self.pin_memory_fn(buffer)
        elif self.device == 'cpu':
            buffer = get_accelerator().pin_memory(buffer, align_bytes=0)

        self.all_buffers[index] = buffer
        self.pin_memory_time_sec += time.time() - start_time
        self.buffer_numel[index] = num_elems

    def _ensure_slot_capacity(self, index, num_elems):
        if self.buffer_numel[index] >= num_elems:
            return

        previous_numel = self.buffer_numel[index]
        self._allocate_slot(index=index, num_elems=num_elems)
        if previous_numel == 0:
            self.num_buffer_allocations += 1
        else:
            self.num_buffer_reallocations += 1

        self.total_bytes = self._pinned_bytes()
        self.gigabytes = self.total_bytes / (1024**3)
        self.max_pinned_bytes = max(self.max_pinned_bytes, self.total_bytes)

        if dist.get_rank() == 0:
            summary = (
                f"SwapBufferManager[{self.name}] pinned slot={index}, "
                f"requested_num_elems={num_elems}, "
                f"previous_num_elems={previous_numel}, "
                f"slot_size={(num_elems * self.element_size) / (1024**3):.2f} GiB, "
                f"pinned_pool_size={self.total_bytes / (1024**3):.2f} GiB, "
                f"max_pool_capacity={self.capacity_bytes / (1024**3):.2f} GiB")
            logger.info(summary)
            print_rank_0(summary, force=True)

    def _select_free_buffer_indices(self, num_elems, count):
        free_indices = self.free_buffer_index
        reusable = sorted([i for i in free_indices if self.buffer_numel[i] >= num_elems],
                          key=lambda i: self.buffer_numel[i])
        unallocated = [i for i in free_indices if self.buffer_numel[i] == 0]
        growable = sorted([i for i in free_indices if 0 < self.buffer_numel[i] < num_elems],
                          key=lambda i: self.buffer_numel[i],
                          reverse=True)
        return (reusable + unallocated + growable)[:count]

    def _pinned_bytes(self):
        return sum(self.buffer_numel) * self.element_size

    def _used_pinned_bytes(self):
        return sum(self.buffer_numel[index] for index in self.used_buffer_index.values()) * self.element_size

    def _used_requested_bytes(self):
        return sum(self.used_buffer_numel.values()) * self.element_size

    def allocate(self, num_elems, count, dtype):
        assert dtype == self.dtype
        assert num_elems <= self.num_elems
        self.max_requested_num_elems = max(self.max_requested_num_elems, num_elems)
        self.max_requested_count = max(self.max_requested_count, count)
        self.max_requested_bytes = max(self.max_requested_bytes, num_elems * count * self.element_size)
        if count <= 0 or count > len(self.free_buffer_index):
            self.num_failed_allocations += 1
            return None

        used_indices = self._select_free_buffer_indices(num_elems=num_elems, count=count)
        assert len(used_indices) == count
        for index in used_indices:
            self._ensure_slot_capacity(index=index, num_elems=num_elems)
        used_index_set = set(used_indices)
        self.free_buffer_index = [i for i in self.free_buffer_index if i not in used_index_set]

        buffers = []
        for i in used_indices:
            tmp_buffer = self.all_buffers[i].narrow(0, 0, num_elems)
            buffers.append(tmp_buffer)
            self.used_buffer_index[id(tmp_buffer)] = i
            self.used_buffer_numel[id(tmp_buffer)] = num_elems
        self.num_allocations += 1
        self.max_allocated_buffers = max(self.max_allocated_buffers, len(self.used_buffer_index))
        self.max_allocated_bytes = max(self.max_allocated_bytes, self._used_pinned_bytes())
        self.max_requested_in_use_bytes = max(self.max_requested_in_use_bytes, self._used_requested_bytes())
        return buffers

    def allocate_lease(self, num_elems, count, dtype, owner):
        buffers = self.allocate(num_elems=num_elems, count=count, dtype=dtype)
        if buffers is None:
            return None
        return SwapBufferLease(manager=self, buffers=buffers, owner=owner)

    def allocate_all(self, num_elems, dtype):
        return self.allocate(num_elems=num_elems, count=len(self.free_buffer_index), dtype=dtype)

    def allocate_all_lease(self, num_elems, dtype, owner):
        return self.allocate_lease(num_elems=num_elems,
                                   count=len(self.free_buffer_index),
                                   dtype=dtype,
                                   owner=owner)

    def free(self, buffers):
        buffer_ids = []
        for buf in buffers:
            buffer_ids.append(id(buf))

        assert all([b_id in self.used_buffer_index for b_id in buffer_ids])

        for b_id in buffer_ids:
            self.free_buffer_index.append(self.used_buffer_index[b_id])
            del (self.used_buffer_index[b_id])
            del (self.used_buffer_numel[b_id])

    def status(self):
        self.total_bytes = self._pinned_bytes()
        self.gigabytes = self.total_bytes / (1024**3)
        self.max_pinned_bytes = max(self.max_pinned_bytes, self.total_bytes)
        return {
            'buffer_num_elems': self.num_elems,
            'name': self.name,
            'lazy': self.lazy,
            'buffer_count': self.count,
            'free_buffer_count': len(self.free_buffer_index),
            'used_buffer_count': len(self.used_buffer_index),
            'element_size': self.element_size,
            'buffer_bytes': self.buffer_bytes,
            'capacity_bytes': self.capacity_bytes,
            'total_bytes': self.total_bytes,
            'pinned_bytes': self.total_bytes,
            'free_bytes': sum(self.buffer_numel[index] for index in self.free_buffer_index) * self.element_size,
            'used_bytes': self._used_pinned_bytes(),
            'used_requested_bytes': self._used_requested_bytes(),
            'pin_memory_time_sec': self.pin_memory_time_sec,
            'num_allocations': self.num_allocations,
            'num_failed_allocations': self.num_failed_allocations,
            'num_buffer_allocations': self.num_buffer_allocations,
            'num_buffer_reallocations': self.num_buffer_reallocations,
            'max_allocated_buffers': self.max_allocated_buffers,
            'max_allocated_bytes': self.max_allocated_bytes,
            'max_requested_in_use_bytes': self.max_requested_in_use_bytes,
            'max_pinned_bytes': self.max_pinned_bytes,
            'max_requested_num_elems': self.max_requested_num_elems,
            'max_requested_count': self.max_requested_count,
            'max_requested_bytes': self.max_requested_bytes,
        }

    def summary(self):
        status = self.status()
        return (
            f"{self.name}: pinned={status['pinned_bytes'] / (1024**3):.2f} GiB, "
            f"capacity={status['capacity_bytes'] / (1024**3):.2f} GiB, "
            f"max_pinned={status['max_pinned_bytes'] / (1024**3):.2f} GiB, "
            f"used_requested={status['used_requested_bytes'] / (1024**3):.2f} GiB, "
            f"max_requested_in_use={status['max_requested_in_use_bytes'] / (1024**3):.2f} GiB, "
            f"buffer_allocations={status['num_buffer_allocations']}, "
            f"buffer_reallocations={status['num_buffer_reallocations']}, "
            f"allocations={status['num_allocations']}, "
            f"failed_allocations={status['num_failed_allocations']}, "
            f"pin_memory_time={status['pin_memory_time_sec']:.3f} sec")

    def allocation_failure_message(self, requested_num_elems, requested_count, owner):
        status = self.status()
        requested_bytes = requested_num_elems * requested_count * self.element_size
        return (
            f"SwapBufferManager[{self.name}] could not allocate {requested_count} buffer(s) for {owner}: "
            f"requested_num_elems={requested_num_elems}, "
            f"requested_bytes={requested_bytes / (1024**3):.2f} GiB, "
            f"free_buffer_count={status['free_buffer_count']}, "
            f"used_buffer_count={status['used_buffer_count']}, "
            f"buffer_count={status['buffer_count']}, "
            f"buffer_num_elems={status['buffer_num_elems']}, "
            f"pinned_pool_size={status['total_bytes'] / (1024**3):.2f} GiB, "
            f"max_pool_capacity={status['capacity_bytes'] / (1024**3):.2f} GiB. "
            "Increase offload_optimizer.buffer_count, reduce zero_optimization.sub_group_size, "
            "or disable optimizer offload pipelining.")


def get_sized_buffer(buffer, num_elems):
    assert num_elems <= buffer.numel(), \
        f'num_elems {num_elems} > buffer {buffer.numel()}'
    return buffer.narrow(0, 0, num_elems) if num_elems < buffer.numel() else buffer


def get_sized_buffers(buffer_list, num_elems_list):
    swap_buffers = [
        get_sized_buffer(buffer, num_elems) \
        for buffer, num_elems in zip(buffer_list, num_elems_list)
    ]
    return swap_buffers
