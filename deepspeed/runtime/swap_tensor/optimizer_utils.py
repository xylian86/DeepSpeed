# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping tensors to/from (NVMe) storage devices.
"""

import os
import torch

from deepspeed import comm as dist
from deepspeed.utils.logging import logger
from deepspeed.runtime.swap_tensor.constants import *
from deepspeed.runtime.swap_tensor.utils import swap_in_tensors, swap_out_tensors, \
    MIN_AIO_BYTES, AIO_ALIGNED_BYTES, get_sized_buffers, is_direct_io_buffer, print_rank_0
from deepspeed.runtime.swap_tensor.utils import SwapBufferManager, SwapBufferPool


OPTIMIZER_SWAP_STAGING_BUFFER_COUNT = 4


def split_swap_buffer_counts(swap_config):
    if not getattr(swap_config, 'pipeline', False):
        return swap_config.buffer_count, 0

    staging_buffer_count = OPTIMIZER_SWAP_STAGING_BUFFER_COUNT
    state_buffer_count = swap_config.buffer_count - staging_buffer_count
    if state_buffer_count <= 0:
        raise ValueError(
            f"Pipeline swap requires more than {staging_buffer_count} total buffers, got {swap_config.buffer_count}")
    return state_buffer_count, staging_buffer_count


class FlattenedTensorSwapInfo(object):

    def __init__(self, path, length, offset, file_offset):
        self.path = path
        self.offset = offset
        self.length = length
        self.file_offset = file_offset


class OptimizerSwapFileAllocator(object):

    def __init__(self, path, element_size, numel_alignment):
        self.path = path
        self.element_size = element_size
        self.numel_alignment = numel_alignment
        self.next_file_offset = 0

    def _aligned_numel(self, numel):
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)

    def allocate(self, numel):
        file_offset = self.next_file_offset
        self.next_file_offset += self._aligned_numel(numel) * self.element_size
        return self.path, file_offset


class SwapTensorContext(object):

    def __init__(self, tensor, swap_allocator, numel):
        self.compute_tensor = tensor
        self.swap_tensor = torch.Tensor()
        self.swap_path, self.swap_offset = swap_allocator.allocate(numel)

    def release_memory(self):
        self.compute_tensor.data = torch.Tensor()
        self.swap_tensor.data = torch.Tensor()

    def set_buffers(self, compute_buffer, swap_buffer):
        self.compute_tensor.data = compute_buffer.data
        self.swap_tensor.data = swap_buffer.data


class OptimizerStateSwapInfo(object):

    def __init__(self, parameter, numel, base_folder, swap_allocator):
        self.tensors = []
        self.param_id = OptimizerSwapper.parameter_id(parameter)
        self.swap_folder = base_folder
        self.swap_allocator = swap_allocator
        self.swapped_gradients = {}
        self.unswapped_gradients = {}
        self.tensor_numel = numel
        self.tensor_dtype = parameter.dtype
        self.tensor_device = parameter.device
        self.has_state_tensors = False
        self.swap_buffers = []
        self.swap_buffer_leases = []
        self._add_tensors([parameter])

    def numel(self):
        return self.tensor_numel

    def has_gradients(self):
        return bool(self.swapped_gradients) or bool(self.unswapped_gradients)

    def _add_tensors(self, tensor_list):
        for t in tensor_list:
            self.tensors.append(SwapTensorContext(t, self.swap_allocator, self.tensor_numel))

    def add_state_tensors(self, tensor_list):
        self.has_state_tensors = True
        self._add_tensors(tensor_list)

    def unbound_direct_tensor_count(self):
        return sum(1 for t in self.tensors if is_direct_io_buffer(t.compute_tensor) and t.swap_tensor.numel() == 0)

    def bind_unbound_direct_swap_buffers(self, buffers, aligned_numel):
        contexts = [t for t in self.tensors if is_direct_io_buffer(t.compute_tensor) and t.swap_tensor.numel() == 0]
        if len(contexts) != len(buffers):
            raise ValueError(f"Expected {len(contexts)} direct swap buffers, got {len(buffers)}")
        if aligned_numel < self.numel():
            raise ValueError(f"Aligned buffer size {aligned_numel} is smaller than tensor size {self.numel()}")
        compute_lengths = [self.numel()] * len(contexts)
        swap_lengths = [aligned_numel] * len(contexts)
        compute_buffers = get_sized_buffers(buffers, compute_lengths)
        swap_buffers = get_sized_buffers(buffers, swap_lengths)

        for context, compute_buffer, swap_buffer in zip(contexts, compute_buffers, swap_buffers):
            compute_buffer.data.copy_(context.compute_tensor.data)
            context.set_buffers(compute_buffer=compute_buffer, swap_buffer=swap_buffer)

    def num_tensors(self):
        return len(self.tensors)

    def device(self):
        return self.tensor_device

    def dtype(self):
        return self.tensor_dtype

    def release_memory(self):
        for t in self.tensors:
            t.release_memory()

    def get_compute_tensors(self):
        return [t.compute_tensor for t in self.tensors]

    def get_swap_paths(self):
        return [t.swap_path for t in self.tensors]

    def get_swap_offsets(self):
        return [t.swap_offset for t in self.tensors]

    def get_swap_buffers_and_paths(self, pinned):
        swap_buffers = []
        swap_paths = []
        swap_offsets = []
        select_tensors = [t for t in self.tensors if is_direct_io_buffer(t.compute_tensor) == pinned]
        for t in select_tensors:
            swap_buffers.append(t.swap_tensor if pinned else t.compute_tensor)
            swap_paths.append(t.swap_path)
            swap_offsets.append(t.swap_offset)
        return swap_buffers, swap_paths, swap_offsets

    def get_or_create_gradient_paths_and_offsets(self, offsets, lengths):
        gradient_paths = []
        gradient_file_offsets = []
        for offset, length in zip(offsets, lengths):
            if offset not in self.swapped_gradients.keys():
                path, file_offset = self.swap_allocator.allocate(length)
                self.swapped_gradients[offset] = FlattenedTensorSwapInfo(path, length, offset, file_offset)

            gradient_paths.append(self.swapped_gradients[offset].path)
            gradient_file_offsets.append(self.swapped_gradients[offset].file_offset)

        return gradient_paths, gradient_file_offsets

    def set_swap_buffers(self, buffers, aligned_numel):
        num_tensors = len(self.tensors)
        compute_lengths = [self.numel()] * num_tensors
        compute_buffers = get_sized_buffers(buffers, compute_lengths)
        swap_lengths = [aligned_numel] * num_tensors
        swap_buffers = get_sized_buffers(buffers, swap_lengths)

        for i, t in enumerate(self.tensors):
            t.set_buffers(compute_buffer=compute_buffers[i], swap_buffer=swap_buffers[i])

    def get_swap_gradient_buffers(self, swap_buffer):
        assert self.numel() <= swap_buffer.numel()
        return [swap_buffer.narrow(0, grad.offset, grad.length) for grad in self.swapped_gradients.values()]

    def get_swap_gradient_paths(self):
        return [grad.path for grad in self.swapped_gradients.values()]

    def get_swap_gradient_offsets(self):
        return [grad.file_offset for grad in self.swapped_gradients.values()]

    def get_unpinned_state_tensors(self):
        return [t.compute_tensor for t in self.tensors if not is_direct_io_buffer(t.compute_tensor)]

    def read_unswapped_gradients(self, dest_buffer):
        num_elem_count = 0
        for offset, grad_partition in self.unswapped_gradients.items():
            dst_tensor = dest_buffer.narrow(0, offset, grad_partition.numel())
            dst_tensor.data.copy_(grad_partition.data)
            num_elem_count += grad_partition.numel()

        return num_elem_count

    def write_unswapped_gradients(self, src_buffer):
        num_elem_count = 0
        for offset, grad_partition in self.unswapped_gradients.items():
            src_tensor = src_buffer.narrow(0, offset, grad_partition.numel())
            grad_partition.data.copy_(src_tensor.data)
            num_elem_count += grad_partition.numel()

        return num_elem_count

    def release_unswapped_gradients(self):
        self.unswapped_gradients = {}


SWAPPER_DEBUG_MODE = False
SWAP_OUT_GRADIENT_TIMER = 'swap_out_gradient'


class OptimizerSwapper(object):

    @staticmethod
    def parameter_id(param):
        return param.ds_id

    def __init__(self,
                 swap_config,
                 aio_config,
                 base_folder,
                 optimizer,
                 largest_numel,
                 device,
                 dtype,
                 timers,
                 buffer_device='cpu',
                 pin_memory_fn=None,
                 unpin_memory_fn=None,
                 lazy_swap_buffers=None):
        self.swap_config = swap_config
        self.aio_config = aio_config

        # NVMe swap management
        self.swap_params_info = {}
        self.swap_element_size = torch.tensor([], dtype=dtype).element_size()
        self.swap_folder = os.path.join(base_folder, 'optimizer', f'rank{dist.get_rank()}')
        os.makedirs(self.swap_folder, exist_ok=True)
        self.swap_container_path = os.path.join(self.swap_folder, 'optimizer_state_and_gradient.swp')

        self.optimizer = optimizer

        # Read/Write alignment for each thread during Intra-request parallelism
        self.min_aio_bytes = max(MIN_AIO_BYTES, aio_config[AIO_BLOCK_SIZE])
        self.aligned_bytes = AIO_ALIGNED_BYTES * aio_config[AIO_INTRA_OP_PARALLELISM]
        self.numel_alignment = self.aligned_bytes // self.swap_element_size
        self.swap_allocator = OptimizerSwapFileAllocator(path=self.swap_container_path,
                                                         element_size=self.swap_element_size,
                                                         numel_alignment=self.numel_alignment)

        # Swap buffer management
        self.largest_numel = self._io_aligned_numel(largest_numel)
        self.dtype = dtype
        state_buffer_count, staging_buffer_count = split_swap_buffer_counts(swap_config)
        lazy_state_buffers = staging_buffer_count > 0 if lazy_swap_buffers is None else bool(lazy_swap_buffers)
        self.staging_num_write_calls = 0
        self.staging_num_chunks_written = 0
        self.staging_num_elements_written = 0
        self.swap_buffer_manager = SwapBufferManager(num_elems=self.largest_numel,
                                                     count=state_buffer_count,
                                                     dtype=dtype,
                                                     name='optimizer_state',
                                                     lazy=lazy_state_buffers,
                                                     device=buffer_device,
                                                     pin_memory_fn=pin_memory_fn,
                                                     unpin_memory_fn=unpin_memory_fn)
        self.staging_swap_buffer_manager = self.swap_buffer_manager
        if staging_buffer_count > 0:
            staging_buffer_numel = self._staging_buffer_numel(swap_config)
            self.staging_swap_buffer_manager = SwapBufferManager(num_elems=staging_buffer_numel,
                                                                 count=staging_buffer_count,
                                                                 dtype=dtype,
                                                                 name='optimizer_staging',
                                                                 device=buffer_device,
                                                                 pin_memory_fn=pin_memory_fn,
                                                                 unpin_memory_fn=unpin_memory_fn)
        self.gradient_buffer_lease = None

        # Timers
        self.timers = timers
        self.timer_names = set()

        # Print exclusion list
        self.print_exclude_list = [
            'optimizer',
            'swap_buffer_manager',
            'staging_swap_buffer_manager',
            'gradient_buffer_lease',
            'swap_params_info',
            'timers',
            'timer_names',
        ]

    def _set_swap_info_lease(self, swap_info, lease):
        swap_info.swap_buffer_leases = [lease]
        swap_info.swap_buffers = lease.buffers.copy()

    def _append_swap_info_lease(self, swap_info, lease):
        swap_info.swap_buffer_leases.append(lease)
        swap_info.swap_buffers += lease.buffers.copy()

    def _release_swap_info_buffers(self, swap_info):
        leases = swap_info.swap_buffer_leases
        if leases:
            for lease in leases:
                lease.release()
            swap_info.swap_buffer_leases = []
        elif swap_info.swap_buffers:
            self.swap_buffer_manager.free(swap_info.swap_buffers)
        swap_info.swap_buffers = []

    def _staging_buffer_numel(self, swap_config):
        configured_numel = max(1, int(getattr(swap_config, 'buffer_size', self.largest_numel)))
        min_aio_numel = max(1, self.min_aio_bytes // self.swap_element_size)
        configured_numel = max(configured_numel, min_aio_numel)
        return min(self.largest_numel, self._io_aligned_numel(configured_numel))

    def _allocate_staging_lease(self, owner):
        manager = self.staging_swap_buffer_manager
        lease = manager.allocate_all_lease(num_elems=manager.num_elems, dtype=self.dtype, owner=owner)
        if lease is None:
            raise RuntimeError(
                manager.allocation_failure_message(requested_num_elems=manager.num_elems,
                                                   requested_count=manager.count,
                                                   owner=owner))
        return lease

    def _split_tensors_for_staging_chunks(self, tensors, offsets):
        chunk_numel = self.staging_swap_buffer_manager.num_elems
        chunked_tensors = []
        chunked_offsets = []

        for tensor, offset in zip(tensors, offsets):
            tensor_numel = tensor.numel()
            tensor_offset = 0
            while tensor_offset < tensor_numel:
                chunk_numel_for_tensor = min(chunk_numel, tensor_numel - tensor_offset)
                chunked_tensors.append(tensor.narrow(0, tensor_offset, chunk_numel_for_tensor))
                chunked_offsets.append(offset + tensor_offset)
                tensor_offset += chunk_numel_for_tensor

        return chunked_tensors, chunked_offsets

    def purge_state(self):
        for swap_info in self.swap_params_info.values():
            swap_info.tensors = [swap_info.tensors[0]]
            swap_info.has_state_tensors = False

    def is_swappable_tensor(self, tensor=None, numel=None):
        assert tensor is not None or numel is not None, "Either tensor or numel must be provided"
        if tensor is not None:
            return self.min_aio_bytes <= (tensor.numel() * self.swap_element_size)
        return self.min_aio_bytes <= (numel * self.swap_element_size)

    def init_timers(self):
        self.timer_names = set()

    def log_timers(self):
        if self.timer_names:
            self._log_timers(list(self.timer_names), force=True)
        self._log_swap_buffer_summary()

    def _log_swap_buffer_summary(self):
        managers = [self.swap_buffer_manager]
        if self.staging_swap_buffer_manager is not self.swap_buffer_manager:
            managers.append(self.staging_swap_buffer_manager)

        summary = "Optimizer swap buffer summary: " + " | ".join([manager.summary() for manager in managers])
        logger.info(summary)
        print_rank_0(summary, force=True)

    def pre_backward(self):
        self.init_timers()

    def post_backward(self):
        pass

    def _flush_gradient_swapper(self, gradient_swapper):
        if gradient_swapper.has_buffers():
            self._start_timer(SWAP_OUT_GRADIENT_TIMER)
            pinned_buffers = gradient_swapper.release_buffers()
            if self.gradient_buffer_lease is not None:
                self.gradient_buffer_lease.release()
                self.gradient_buffer_lease = None
            else:
                self.staging_swap_buffer_manager.free(pinned_buffers)
            self._stop_timer(SWAP_OUT_GRADIENT_TIMER)
            self.timer_names.add(SWAP_OUT_GRADIENT_TIMER)
            self.timer_names.update(gradient_swapper.get_timer_names())

    def _swap_out_gradients(self, parameter, gradient_offsets, gradient_tensors, gradient_swapper):
        if OptimizerSwapper.parameter_id(parameter) not in self.swap_params_info.keys():
            return

        swap_info = self.swap_params_info[OptimizerSwapper.parameter_id(parameter)]

        swappable_tensors = []
        swappable_offsets = []
        swappable_lengths = []

        aligned_gradients, aligned_offsets = self._adjust_for_misaligned_lengths(tensors=gradient_tensors,
                                                                                 offsets=gradient_offsets)

        self._start_timer(SWAP_OUT_GRADIENT_TIMER)
        for tensor, offset in zip(aligned_gradients, aligned_offsets):
            if not self.is_swappable_tensor(tensor=tensor):
                swap_info.unswapped_gradients[offset] = tensor
                continue

            swappable_tensors.append(tensor)
            swappable_offsets.append(offset)
            swappable_lengths.append(tensor.numel())

        if len(swappable_tensors) > 0:
            if not gradient_swapper.has_buffers():
                lease = self._allocate_staging_lease(owner='gradient swap-out staging')

                self.gradient_buffer_lease = lease
                try:
                    gradient_swapper.add_buffers(lease.buffers)
                except Exception:
                    self.gradient_buffer_lease = None
                    lease.release()
                    raise

            swappable_tensors, swappable_offsets = self._split_tensors_for_staging_chunks(swappable_tensors,
                                                                                          swappable_offsets)
            swappable_lengths = [tensor.numel() for tensor in swappable_tensors]
            swappable_paths, swappable_file_offsets = swap_info.get_or_create_gradient_paths_and_offsets(
                swappable_offsets, swappable_lengths)

            gradient_swapper.swap_out_tensors(tensor_list=swappable_tensors,
                                              path_list=swappable_paths,
                                              offset_list=swappable_file_offsets)

        self._stop_timer(SWAP_OUT_GRADIENT_TIMER)
        self.timer_names.add(SWAP_OUT_GRADIENT_TIMER)

    def _initialize_from_swapped_fp16_params(self, aio_handle, fp16_partitions_info, fp16_num_elems,
                                             fp16_pinned_buffers, fp32_parameters):
        assert len(fp32_parameters) == len(fp16_partitions_info)
        assert len(fp32_parameters) == len(fp16_num_elems)
        assert all([is_direct_io_buffer(buffer) for buffer in fp16_pinned_buffers])

        fp32_swap_paths, fp32_swap_offsets = self._get_swap_paths_and_offsets(parameters=fp32_parameters,
                                                                              num_elems=fp16_num_elems)

        fp32_pinned_lease = self._allocate_staging_lease(owner='initialize swapped fp16 params')
        fp32_pinned_buffers = fp32_pinned_lease.buffers

        try:
            fp16_buffer_numel = [buf.numel() for buf in fp16_pinned_buffers]
            assert all([numel >= self.largest_numel for numel in fp16_buffer_numel]), \
            f"numel of fp16 buffers {fp16_buffer_numel} is too small for initializing fp32 params {self.largest_numel}"

            fp16_swap_buffers = SwapBufferPool(fp16_pinned_buffers)

            curr_index = 0
            while curr_index < len(fp32_parameters):
                fp16_pinned_tensors = self._swap_in_fp16_params(aio_handle=aio_handle,
                                                                fp16_num_elems=fp16_num_elems[curr_index:],
                                                                fp16_partitions_info=fp16_partitions_info[curr_index:],
                                                                fp16_swap_buffers=fp16_swap_buffers)

                if dist.get_rank() == 0 and SWAPPER_DEBUG_MODE:
                    for i, tensor in enumerate(fp16_pinned_tensors):
                        true_index = curr_index + i
                        logger.info(
                            f'swap_in_fp16_param: fp32_id = {OptimizerSwapper.parameter_id(fp32_parameters[true_index])} index = {true_index} orig_num_elem = {fp16_num_elems[true_index]}, swap_num_elem = {fp16_pinned_tensors[i].numel()}'
                        )

                swap_out_count = self._swap_out_unpinned_tensors(aio_handle=aio_handle,
                                                                 unpinned_tensors=fp16_pinned_tensors,
                                                                 dest_paths=fp32_swap_paths[curr_index:],
                                                                 dest_offsets=fp32_swap_offsets[curr_index:],
                                                                 pinned_buffers=fp32_pinned_buffers)
                assert swap_out_count == len(fp16_pinned_tensors), \
                f"{swap_out_count} does not match {len(fp16_pinned_tensors)}"

                fp16_swap_buffers.reset()
                curr_index += swap_out_count

        finally:
            fp32_pinned_lease.release()

    def _swap_in_fp16_params(self, aio_handle, fp16_num_elems, fp16_partitions_info, fp16_swap_buffers):
        assert len(fp16_num_elems) > 0

        swapped_fp16_tensors = []
        swap_tensors = []
        swap_paths = []
        unswapped_srcs = []
        unswapped_dsts = []

        for i, numel in enumerate(fp16_num_elems):
            pinned_tensor, _ = fp16_swap_buffers.allocate_tensor(numel, None, numel)
            if pinned_tensor is None:
                break

            swapped_fp16_tensors.append(pinned_tensor)
            offset = 0
            for tensor, partition_numel, partition_path in fp16_partitions_info[i]:
                dst_tensor = pinned_tensor.narrow(0, offset, partition_numel)
                if partition_path is None:
                    unswapped_srcs.append(tensor)
                    unswapped_dsts.append(dst_tensor)
                else:
                    swap_paths.append(partition_path)
                    swap_tensors.append(dst_tensor)
                offset += partition_numel

        assert len(swapped_fp16_tensors) + len(unswapped_srcs) > 0
        num_swap_ops = swap_in_tensors(aio_handle, swap_tensors, swap_paths)
        for src, dst in zip(unswapped_srcs, unswapped_dsts):
            dst.data.copy_(src.data)

        if len(swap_tensors) > 0:
            assert num_swap_ops == aio_handle.wait()

        return swapped_fp16_tensors

    def _swap_out_fp16_params(self,
                              aio_handle,
                              fp32_swap_paths,
                              fp32_swap_buffers,
                              fp16_pinned_tensors,
                              fp32_swap_offsets=None):
        assert len(fp16_pinned_tensors) <= len(fp32_swap_paths)
        if fp32_swap_offsets is None:
            fp32_swap_offsets = [0] * len(fp16_pinned_tensors)
        assert len(fp16_pinned_tensors) <= len(fp32_swap_offsets)

        swap_out_count = 0
        for i, fp16_tensor in enumerate(fp16_pinned_tensors):
            if not fp32_swap_buffers.has_space(fp16_tensor.numel()):
                fp32_swap_buffers.swap_out(aio_handle)
                fp32_swap_buffers.reset()

            pinned_tensor, _ = fp32_swap_buffers.insert_tensor(fp16_tensor, fp32_swap_paths[i],
                                                               self._io_aligned_numel(fp16_tensor.numel()),
                                                               fp32_swap_offsets[i])
            assert pinned_tensor is not None
            swap_out_count += 1

        if len(fp32_swap_buffers.get_swap_tensors()) > 0:
            fp32_swap_buffers.swap_out(aio_handle)

        return swap_out_count

    def _initialize_parameters(self, parameters, src_tensors, aio_handle):
        assert len(parameters) == len(src_tensors)

        swap_paths, swap_offsets = self._get_swap_paths_and_offsets(parameters=parameters,
                                                                    num_elems=[src.numel() for src in src_tensors])

        SWAP_INIT_TIMER = "swap_init_write"
        self._start_timer(SWAP_INIT_TIMER)

        pinned_lease = self._allocate_staging_lease(owner='initialize optimizer params')
        pinned_buffers = pinned_lease.buffers

        try:
            self._swap_out_unpinned_tensors(aio_handle=aio_handle,
                                            unpinned_tensors=src_tensors,
                                            dest_paths=swap_paths,
                                            dest_offsets=swap_offsets,
                                            pinned_buffers=pinned_buffers)

            if dist.get_rank() == 0 and SWAPPER_DEBUG_MODE:
                for i, tensor in enumerate(src_tensors):
                    logger.info(
                        f'copy_in_fp16_param: fp32_id = {OptimizerSwapper.parameter_id(parameters[i])} index = {i}, swap_num_elem = {src_tensors[i].numel()}'
                    )

        finally:
            pinned_lease.release()

        self._stop_timer(SWAP_INIT_TIMER)
        self._log_timers([SWAP_INIT_TIMER])

    def _get_swap_paths_and_offsets(self, parameters, num_elems):
        swap_info_list = [
            self._create_param_swap_info(parameter=p,
                                         numel=numel) \
            for p, numel in zip(parameters, num_elems)
        ]
        assert len(swap_info_list) == len(num_elems)

        swap_paths = [info.tensors[0].swap_path for info in swap_info_list]
        swap_offsets = [info.tensors[0].swap_offset for info in swap_info_list]
        return swap_paths, swap_offsets

    def _get_swap_paths(self, parameters, num_elems):
        swap_paths, _ = self._get_swap_paths_and_offsets(parameters, num_elems)
        return swap_paths

    def _swap_out_unpinned_tensors(self, aio_handle, unpinned_tensors, dest_paths, pinned_buffers, dest_offsets=None):
        assert len(unpinned_tensors) <= len(dest_paths)
        assert len(pinned_buffers) > 0
        if dest_offsets is None:
            dest_offsets = [0] * len(unpinned_tensors)
        assert len(unpinned_tensors) <= len(dest_offsets)

        pending_swap_count = 0
        buffer_index = 0
        chunks_written = 0
        elements_written = 0

        def wait_for_pending_writes():
            nonlocal pending_swap_count, buffer_index
            if pending_swap_count > 0:
                assert aio_handle.wait() == pending_swap_count
                pending_swap_count = 0
                buffer_index = 0

        for src_tensor, dest_path, dest_offset in zip(unpinned_tensors, dest_paths, dest_offsets):
            aligned_numel = self._io_aligned_numel(src_tensor.numel())
            tensor_offset = 0

            while tensor_offset < aligned_numel:
                if buffer_index == len(pinned_buffers):
                    wait_for_pending_writes()

                staging_buffer = pinned_buffers[buffer_index]
                chunk_numel = min(staging_buffer.numel(), aligned_numel - tensor_offset)
                compute_numel = min(chunk_numel, max(0, src_tensor.numel() - tensor_offset))

                if compute_numel > 0:
                    dst_tensor = staging_buffer.narrow(0, 0, compute_numel)
                    src_slice = src_tensor.narrow(0, tensor_offset, compute_numel)
                    dst_tensor.data.copy_(src_slice.data)

                swap_buffer = staging_buffer.narrow(0, 0, chunk_numel)
                swap_offset = dest_offset + tensor_offset * self.swap_element_size
                pending_swap_count += swap_out_tensors(aio_handle, [swap_buffer], [dest_path], [swap_offset])
                buffer_index += 1
                chunks_written += 1
                elements_written += chunk_numel
                tensor_offset += chunk_numel

        wait_for_pending_writes()
        self.staging_num_write_calls += 1
        self.staging_num_chunks_written += chunks_written
        self.staging_num_elements_written += elements_written

        if dist.get_rank() == 0:
            logger.debug(
                f"optimizer staging swap-out: tensors={len(unpinned_tensors)}, "
                f"chunks={chunks_written}, "
                f"bytes={(elements_written * self.swap_element_size) / (1024**3):.2f} GiB")

        return len(unpinned_tensors)

    def _adjust_for_misaligned_lengths(self, tensors, offsets):
        new_tensors = []
        new_offsets = []

        for orig_tensor, orig_offset in zip(tensors, offsets):
            if not self.is_swappable_tensor(tensor=orig_tensor):
                new_tensors.append(orig_tensor)
                new_offsets.append(orig_offset)
                continue

            remainder = orig_tensor.numel() % self.numel_alignment
            if remainder == 0:
                new_tensors.append(orig_tensor)
                new_offsets.append(orig_offset)
                continue

            # Split into two by making remainder a tensor
            aligned_length = (orig_tensor.numel() // self.numel_alignment) * self.numel_alignment
            new_tensors.append(orig_tensor.narrow(0, 0, aligned_length))
            new_offsets.append(orig_offset)

            # remainder tensor
            new_tensors.append(orig_tensor.narrow(0, aligned_length, remainder))
            new_offsets.append(orig_offset + aligned_length)

        return new_tensors, new_offsets

    def _retrieve_unswapped_grad_partitions(self, swap_info, dest_buffer):
        UNSWAPPED_READ_GRADIENTS = 'unswapped_read_gradients'
        self._start_timer(UNSWAPPED_READ_GRADIENTS)
        tensor_count = len(swap_info.unswapped_gradients)
        num_elem_count = swap_info.read_unswapped_gradients(dest_buffer)
        self._stop_timer(UNSWAPPED_READ_GRADIENTS)
        self._log_timers([UNSWAPPED_READ_GRADIENTS])

        # It should be safe to discard unswapped gradient partitions
        swap_info.release_unswapped_gradients()

        if SWAPPER_DEBUG_MODE:
            logger.info(
                f'optimizer_retrieve_unswapped_gradients: param={swap_info.param_id} tensor_count={tensor_count} elem_count={num_elem_count}'
            )

    def _get_state_tensors(self, parameter):
        if parameter not in self.optimizer.state:
            return []

        tensor_list = []
        for state_name, value in self.optimizer.state[parameter].items():
            if torch.is_tensor(value) and self.is_swappable_tensor(tensor=value):
                value.ds_id = state_name + '-' + parameter.ds_id
                tensor_list.append(value)

        return tensor_list

    def _update_param_state_info(self, swap_info, parameter):
        if not swap_info.has_state_tensors:
            state_tensors = self._get_state_tensors(parameter)
            if state_tensors:
                swap_info.add_state_tensors(state_tensors)

    def _create_param_swap_info(self, parameter, numel):
        param_id = OptimizerSwapper.parameter_id(parameter)
        assert param_id not in self.swap_params_info

        self.swap_params_info[param_id] = OptimizerStateSwapInfo(parameter=parameter,
                                                                 numel=numel,
                                                                 base_folder=self.swap_folder,
                                                                 swap_allocator=self.swap_allocator)
        swap_info = self.swap_params_info[param_id]

        self._update_param_state_info(swap_info, parameter)

        return swap_info

    def _get_param_swap_info(self, parameter):
        param_id = OptimizerSwapper.parameter_id(parameter)
        swap_info = self.swap_params_info.get(param_id, None)

        if swap_info is not None:
            self._update_param_state_info(swap_info, parameter)

        return swap_info

    def _start_timer(self, name):
        if self.timers:
            self.timers(name).start()

    def _stop_timer(self, name):
        if self.timers:
            self.timers(name).stop()

    def _log_timers(self, name_list, force=False):
        if self.timers and (SWAPPER_DEBUG_MODE or force):
            self.timers.log(name_list)

    def _io_aligned_numel(self, numel):
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)
