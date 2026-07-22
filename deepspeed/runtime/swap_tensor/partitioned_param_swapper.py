# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping tensors to/from (NVMe) storage devices.
"""

import os
import shutil
from enum import Enum
import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from .constants import *
from .utils import swap_in_tensors, swap_out_tensors, MIN_AIO_BYTES, AIO_ALIGNED_BYTES, print_object, SwapBufferPool, \
    get_swap_io_handle_factory
from .lookahead_dram_cache import LookaheadDRAMCache
from deepspeed.runtime.zero.offload_config import resolve_nvme_path


def print_rank_0(message, debug=False, force=False):
    if dist.get_rank() == 0 and (debug or force):
        print(message)


class PartitionedParamStatus(Enum):
    # Partitioned parameters are present and ready for use
    AVAILABLE = 1

    # partitioned params are in some non-memory device
    NOT_AVAILABLE = 2

    # partitioned params are being read from some non-memory device.
    INFLIGHT = 3


class AsyncPartitionedParameterSwapper(object):

    def __init__(self, ds_config, model_dtype):

        self.dtype = model_dtype

        #set swap buffers, create aio handles
        self._configure_aio(ds_config)

        #mapping from param id to path
        self.id_to_path = {}

        #mapping from pram_id to buffer id
        self.param_id_to_buffer_id = {}

        # mapping from param_id to swap buffer
        self.param_id_to_swap_buffer = {}

        #number of elements in the param
        self.param_id_to_numel = {}

        self.pending_writes = 0
        self.pending_reads = 0

        #keep track of async swap in params and buffers
        self.inflight_params = []
        self.inflight_swap_in_buffers = []
        self.inflight_numel = 0

        #keep track of available params
        self.available_params = set()
        self.available_param_objects = {}
        self.available_numel = 0

        # for swapping out from partitioned fp32 params
        self.partitioned_swap_buffer = None
        self.partitioned_swap_pool = None

        self.invalid_buffer = torch.tensor(1).half()

        if dist.get_rank() == 0:
            exclude_list = ['aio_read_handle', 'aio_write_handle', 'buffers']
            print_object(obj=self, name='AsyncPartitionedParameterSwapper', exclude_list=exclude_list)

    def available_swap_in_buffers(self):
        return len(self.available_buffer_ids)

    def _configure_aio(self, ds_config):
        self.swap_config = ds_config.zero_config.offload_param
        torch_dtype_string = str(self.dtype).split(".")[1]
        nvme_path = resolve_nvme_path(self.swap_config)
        if nvme_path is None:
            raise ValueError("ZeRO-3 NVMe parameter offload requires nvme_path or nvme_path_per_local_rank.")
        self.swap_folder = os.path.join(nvme_path, 'zero_stage_3', f'{torch_dtype_string}params',
                                        f'rank{dist.get_rank()}')
        shutil.rmtree(self.swap_folder, ignore_errors=True)
        os.makedirs(self.swap_folder, exist_ok=True)

        self.swap_element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.aio_config = ds_config.aio_config

        self.use_gds = self.aio_config[AIO_USE_GDS]
        self.aio_handle = get_swap_io_handle_factory(self.aio_config, use_gds=self.use_gds, verbose=False)

        # Read/Write alignment for each thread during Intra-request parallelism
        self.min_aio_bytes = max(MIN_AIO_BYTES, self.aio_config[AIO_BLOCK_SIZE])
        self.aligned_bytes = AIO_ALIGNED_BYTES * self.aio_config[AIO_INTRA_OP_PARALLELISM]
        self.numel_alignment = self.aligned_bytes // self.swap_element_size

        self.elements_per_buffer = self.swap_config.buffer_size
        self.aligned_elements_per_buffer = self._io_aligned_numel(self.elements_per_buffer)
        self.param_buffer_count = self.swap_config.buffer_count

        self.available_buffer_ids = [i for i in range(self.param_buffer_count)]
        self.reserved_buffer_ids = []

        self.aio_read_handle = self.aio_handle(block_size=self.aio_config[AIO_BLOCK_SIZE],
                                               queue_depth=self.aio_config[AIO_QUEUE_DEPTH],
                                               single_submit=self.aio_config[AIO_SINGLE_SUBMIT],
                                               overlap_events=self.aio_config[AIO_OVERLAP_EVENTS],
                                               intra_op_parallelism=self.aio_config[AIO_INTRA_OP_PARALLELISM])

        self.aio_write_handle = self.aio_handle(block_size=self.aio_config[AIO_BLOCK_SIZE],
                                                queue_depth=self.aio_config[AIO_QUEUE_DEPTH],
                                                single_submit=self.aio_config[AIO_SINGLE_SUBMIT],
                                                overlap_events=self.aio_config[AIO_OVERLAP_EVENTS],
                                                intra_op_parallelism=self.aio_config[AIO_INTRA_OP_PARALLELISM])

        self.lookahead_cache = self._configure_lookahead_cache(ds_config)

        buffer_device = get_accelerator().device_name() if self.use_gds else "cpu"
        self.buffers = torch.empty(int(self.aligned_elements_per_buffer * self.param_buffer_count),
                                   dtype=self.dtype,
                                   device=buffer_device,
                                   requires_grad=False)
        if self.use_gds:
            self.aio_read_handle.pin_device_tensor(self.buffers)
        else:
            self.buffers = get_accelerator().pin_memory(self.buffers, align_bytes=0)

        self.swap_out_params = []

    def _configure_lookahead_cache(self, ds_config):
        cache_config = getattr(ds_config, "superrl_cache_config", None)
        if not getattr(cache_config, "enabled", False):
            return None

        aio_handle_factory = get_swap_io_handle_factory(self.aio_config, use_gds=False, verbose=False)
        aio_read_handle = aio_handle_factory(block_size=self.aio_config[AIO_BLOCK_SIZE],
                                             queue_depth=self.aio_config[AIO_QUEUE_DEPTH],
                                             single_submit=self.aio_config[AIO_SINGLE_SUBMIT],
                                             overlap_events=self.aio_config[AIO_OVERLAP_EVENTS],
                                             intra_op_parallelism=self.aio_config[AIO_INTRA_OP_PARALLELISM])
        cache = LookaheadDRAMCache(enabled=True,
                                   dtype=self.dtype,
                                   aio_handle=aio_read_handle,
                                   pin_memory_fn=get_accelerator().pin_memory,
                                   aligned_numel_fn=self._io_aligned_numel,
                                   max_prefetches_per_call=max(1, self.aio_config[AIO_QUEUE_DEPTH]),
                                   max_inflight_prefetches=max(1, 2 * self.aio_config[AIO_QUEUE_DEPTH]))
        print_rank_0(
            f"SuperRL-Cache enabled for NVMe params, host memory envelope={cache.memory_cap_bytes / (1024**3):.2f} GiB",
            force=True)
        return cache

    def cache_enabled(self):
        return self.lookahead_cache is not None and self.lookahead_cache.enabled

    def prefetch_cache_for_trace(self, ordered_params):
        if not self.cache_enabled():
            return

        cache_params = [param for param in ordered_params if getattr(param, "nvme_swapper", None) is self]
        self.lookahead_cache.prefetch(cache_params, path_fn=lambda param: self.get_path(param, must_exist=True))

    def log_cache_stats(self, trace_len=None):
        if not self.cache_enabled():
            return
        if trace_len is not None:
            self.lookahead_cache.trace_len = max(self.lookahead_cache.trace_len, trace_len)
        print_rank_0(str(self.lookahead_cache.stats()), force=True)

    def invalidate_cache(self, params):
        if self.cache_enabled():
            self.lookahead_cache.invalidate(params)

    def _mark_params_available(self, params, numel=None):
        if not hasattr(self, "available_param_objects"):
            self.available_param_objects = {}
        if numel is None:
            numel = sum(self.param_id_to_numel[param.ds_id] for param in params)
        newly_available = []
        for param in params:
            if param.ds_id not in self.available_params:
                self.available_params.add(param.ds_id)
                newly_available.append(param)
            self.available_param_objects[param.ds_id] = param
        if len(newly_available) == len(params):
            self.available_numel += numel
        else:
            self.available_numel += sum(self.param_id_to_numel[param.ds_id] for param in newly_available)

    def _mark_params_not_available(self, params):
        if not hasattr(self, "available_param_objects"):
            self.available_param_objects = {}
        for param in params:
            param_id = param.ds_id
            if param_id in self.available_params:
                self.available_params.remove(param_id)
                self.available_numel -= self.param_id_to_numel[param_id]
            self.available_param_objects.pop(param_id, None)

    def release_available_parameters(self, count, exclude_param_ids=None):
        if count <= 0:
            return 0
        if exclude_param_ids is None:
            exclude_param_ids = set()
        else:
            exclude_param_ids = set(exclude_param_ids)
        if not hasattr(self, "available_param_objects"):
            self.available_param_objects = {}

        released = 0
        for param_id in list(self.available_params):
            if param_id in exclude_param_ids:
                continue
            param = self.available_param_objects.get(param_id)
            if param is None:
                continue
            active_sub_modules = getattr(param, "ds_active_sub_modules", None)
            if active_sub_modules:
                continue

            self.remove_partition_and_release_buffers([param])
            released += 1
            if released >= count:
                break
        return released

    def _swap_in_from_cache(self, params):
        if not self.cache_enabled():
            return params

        cache_misses = []
        for param in params:
            compute_buffer = self.lookahead_cache.acquire(param)
            if compute_buffer is None:
                cache_misses.append(param)
                continue

            self.param_id_to_numel[param.ds_id] = param.ds_tensor.ds_numel
            param.ds_tensor.data = compute_buffer.data
            param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
            self._mark_params_available([param], numel=compute_buffer.numel())
        return cache_misses

    #Check if partitioned param or numel in a tensor is swappable or not
    def swappable_tensor(self, param=None, numel=None):
        if param is not None:
            assert numel is None, "Both parma and numel cannot be provided"
            numel = param.ds_tensor.ds_numel
        if numel is not None:
            return self.min_aio_bytes <= numel * self.swap_element_size
        assert False, "Either param or numel must be provided"

    def get_path(self, param, must_exist=False):
        paths = self._get_swap_paths([param], must_exist=must_exist)
        return paths[0]

    def _get_swap_paths(self, params, must_exist=False):
        paths = []
        for param in params:
            param_id = param.ds_id
            if param_id in self.id_to_path.keys():
                param_path = self.id_to_path[param_id]
            else:
                assert not must_exist, f"Path for param id {param_id} does not exist"
                param_path = os.path.join(self.swap_folder, f'{param_id}_param.tensor.swp')

                self.id_to_path[param_id] = param_path
            paths.append(param_path)

        return paths

    def _get_swap_buffers(self, params):
        buffers = []
        for param in params:
            param_id = param.ds_id
            assert param_id in self.param_id_to_swap_buffer.keys(), \
            f'param {param_id} has not been assigned a swap buffer'
            buffers.append(self.param_id_to_swap_buffer[param_id])

        return buffers

    def _track_numel(self, params):
        for param in params:
            assert param.ds_tensor is not None, "Partitioned tensor is None"
            self.param_id_to_numel[param.ds_id] = param.ds_tensor.ds_numel

    def _allocate_and_return_buffers_for_swap_in(self, params):
        compute_buffers = []
        swap_buffers = []

        for param in params:
            param_id = param.ds_id
            assert param_id in self.param_id_to_numel.keys(), f" Number of elements in param {param_id} is unknown"
            assert param_id not in self.param_id_to_buffer_id.keys(
            ), f"param {param_id} already assigned swap buffer id {self.param_id_to_buffer_id[param_id]}"
            assert param_id not in self.param_id_to_swap_buffer.keys(
            ), f"param {param_id} has already been assigned a swap buffer"

            buffer_id = self.available_buffer_ids.pop()
            print_rank_0(f"param {param.ds_id} is assigned swap in buffer id {buffer_id}  ")
            self.param_id_to_buffer_id[param_id] = buffer_id
            aligned_swap_numel = self._io_aligned_numel(self.param_id_to_numel[param_id])
            swap_buffer = self.buffers.narrow(0, int(buffer_id * self.aligned_elements_per_buffer), aligned_swap_numel)

            self.param_id_to_swap_buffer[param_id] = swap_buffer
            compute_buffer = swap_buffer.narrow(0, 0, self.param_id_to_numel[param_id])
            compute_buffers.append(compute_buffer)
            swap_buffers.append(swap_buffer)

        return compute_buffers, swap_buffers

    #waits for inflight nvme write to complete
    def synchronize_writes(self):
        if self.pending_writes == 0:
            return
        assert self.pending_writes == self.aio_write_handle.wait()
        self.pending_writes = 0
        self.remove_partition_and_release_buffers(self.swap_out_params)
        self.swap_out_params = []

    #waits for inflight nvme reads to complete
    def synchronize_reads(self):
        if self.pending_reads == 0:
            return

        assert self.pending_reads == self.aio_read_handle.wait()

        self.pending_reads = 0

        for param, swap_in_buffer in zip(self.inflight_params, self.inflight_swap_in_buffers):
            param_id = param.ds_id
            compute_buffer = swap_in_buffer.narrow(0, 0, self.param_id_to_numel[param_id])
            param.ds_tensor.data = compute_buffer.data
            param.ds_tensor.status = PartitionedParamStatus.AVAILABLE

        self._mark_params_available(self.inflight_params, numel=self.inflight_numel)

        self.inflight_params = []
        self.inflight_swap_in_buffers = []
        self.inflight_numel = 0

    #Removes the memory assignment and releases the buffers
    #Should only be executed after swapping out the tensors
    def remove_partition_and_release_buffers(self, params):
        for param in params:
            param_id = param.ds_id

            if param_id in self.param_id_to_buffer_id.keys():

                buffer_id = self.param_id_to_buffer_id[param_id]

                assert buffer_id is not None, "Missing buffer id for releasing"

                self.available_buffer_ids.append(buffer_id)
                del self.param_id_to_buffer_id[param_id]
                del self.param_id_to_swap_buffer[param_id]
                print_rank_0(f"param {param.ds_id} releases buffer id {buffer_id}  ")

            self._mark_params_not_available([param])
            param.ds_tensor.data = self.invalid_buffer.data
            param.ds_tensor.status = PartitionedParamStatus.NOT_AVAILABLE
            if self.cache_enabled():
                self.lookahead_cache.on_param_detached(param)

    #writes from in memory to nvme. Does not release the buffers
    def _swap_out(self, params, async_op=True):
        self.invalidate_cache(params)

        swap_out_paths = self._get_swap_paths(params)
        swap_out_params = self._get_swap_buffers(params)
        self._track_numel(params)

        num_swap_ops = swap_out_tensors(self.aio_write_handle, swap_out_params, swap_out_paths)

        self.pending_writes += num_swap_ops
        self.swap_out_params += params

        if not async_op:
            self.synchronize_writes()

    #blocking swap out followed by releasing the memory buffers
    def swap_out_and_release(self, params, async_op=False, force_buffer_release=False):
        if async_op:
            assert force_buffer_release, "Should not release preallocated buffers without completing the swap out. Set force_buffer_release to True to do it anyways"
        self._swap_out(params, async_op=async_op)

    # book keeping function for inflight swap in
    def _update_inflight_swap_in(self, params, swap_in_buffers, inflight_numel, num_swap_ops=None):
        self.inflight_params.extend(params)
        self.inflight_swap_in_buffers.extend(swap_in_buffers)
        self.inflight_numel += inflight_numel

        for param in params:
            param.ds_tensor.status = PartitionedParamStatus.INFLIGHT

        self.pending_reads += num_swap_ops if num_swap_ops is not None else len(params)

    #assigns an in memory buffer and swaps in from nvme
    def swap_in(self, params, async_op=True, swap_in_buffers=None):
        if swap_in_buffers is None:
            deduped_params = []
            seen_param_ids = set()
            for param in params:
                if param.ds_id in seen_param_ids:
                    continue
                seen_param_ids.add(param.ds_id)
                deduped_params.append(param)
            params = deduped_params
            if not params:
                return

        assert all([param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE
                    for param in params]), "Some params are already available or in flight"
        if swap_in_buffers is None:
            params = self._swap_in_from_cache(params)
            if not params:
                return

        swap_in_paths = self._get_swap_paths(params)

        if swap_in_buffers is None:
            if len(self.available_buffer_ids) < len(swap_in_paths):
                needed_buffers = len(swap_in_paths) - len(self.available_buffer_ids)
                self.release_available_parameters(needed_buffers, exclude_param_ids=[p.ds_id for p in params])

            if len(self.available_buffer_ids) < len(swap_in_paths):
                ids = [p.ds_id for p in params]
                print_rank_0(
                    f'Not enough swap in buffers {len(self.available_buffer_ids)} for {len(swap_in_paths)} params, ids = {ids}',
                    force=True)
                print_rank_0(
                    f'Num inflight: params {len(self.inflight_params)}, buffers {len(self.inflight_swap_in_buffers)}, numel = {self.inflight_numel}',
                    force=True)
                print_rank_0(
                    f'Num available params: count = {len(self.available_params)}, ids = {self.available_params}, numel = {self.available_numel}',
                    force=True)

            assert len(swap_in_paths) <= len(
                self.available_buffer_ids
            ), f"Not enough buffers {len(self.available_buffer_ids)} for swapping {len(swap_in_paths)}"
            compute_buffers, swap_in_buffers = self._allocate_and_return_buffers_for_swap_in(params)
            inflight_numel = sum([t.numel() for t in compute_buffers])
        else:
            inflight_numel = sum([t.numel() for t in swap_in_buffers])

        num_swap_ops = swap_in_tensors(self.aio_read_handle, swap_in_buffers, swap_in_paths)

        self._update_inflight_swap_in(params, swap_in_buffers, inflight_numel, num_swap_ops)

        if not async_op:
            self.synchronize_reads()

    # Enables swapping into buffer that is out the control of swapper. This is always synchronous
    def swap_into_buffer(self, param, dest_buffer):
        assert param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE, f"param {param.ds_id} is already available or inflight"

        require_swap_buffer = not (get_accelerator().is_pinned(dest_buffer)
                                   and self._is_io_aligned(dest_buffer.numel()))

        if require_swap_buffer:
            assert len(self.available_buffer_ids) > 0, f"No buffer available to swap param {param.ds_id}."
            compute_buffers, swap_in_buffers = self._allocate_and_return_buffers_for_swap_in([param])
            inflight_numel = compute_buffers[0].numel()
        else:
            swap_in_buffers = [dest_buffer]
            inflight_numel = dest_buffer.numel()

        swap_in_paths = self._get_swap_paths([param])

        num_swap_ops = swap_in_tensors(self.aio_read_handle, swap_in_buffers, swap_in_paths)
        self._update_inflight_swap_in([param], swap_in_buffers, inflight_numel, num_swap_ops)
        self.synchronize_reads()

        if require_swap_buffer:
            dest_buffer.data.copy_(param.ds_tensor.data)
            # Release swap buffer memory assignment. Note, this will mark the parameter not available.
            self.remove_partition_and_release_buffers([param])

    #assign a buffer to a param and return the buffer
    def get_buffer(self, param, numel):
        param_id = param.ds_id

        assert self.available_swap_in_buffers(
        ) > 0, f"No swap buffers to allocate for fp16 param {param_id} of numel = {numel}"
        assert numel <= self.elements_per_buffer, f"More elements {numel} than buffer size {self.elements_per_buffer}"

        self.param_id_to_numel[param_id] = numel
        buffer_id = self.available_buffer_ids.pop()
        self.param_id_to_buffer_id[param_id] = buffer_id
        aligned_swap_numel = self._io_aligned_numel(self.param_id_to_numel[param_id])
        swap_buffer = self.buffers.narrow(0, int(buffer_id * self.aligned_elements_per_buffer), aligned_swap_numel)

        self.param_id_to_swap_buffer[param_id] = swap_buffer
        compute_buffer = swap_buffer.narrow(0, 0, self.param_id_to_numel[param_id])
        print_rank_0(f"param {param.ds_id} is assigned swap in buffer id {buffer_id}")
        return compute_buffer

    def reserve_available_buffers(self):
        buffers = []
        for id in self.available_buffer_ids:
            buffers.append(
                self.buffers.narrow(0, int(id * self.aligned_elements_per_buffer),
                                    int(self.aligned_elements_per_buffer)))
            self.reserved_buffer_ids.append(id)

        self.available_buffer_ids = []
        return buffers

    def release_reserved_buffers(self):
        for id in self.reserved_buffer_ids:
            self.available_buffer_ids.append(id)
        self.reserved_buffer_ids = []

    def _io_aligned_numel(self, numel):
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)

    def _is_io_aligned(self, numel):
        return (numel % self.numel_alignment) == 0

    def reserve_partitioned_swap_space(self, partition_num_elems):
        aligned_numel = sum([self._io_aligned_numel(numel) for numel in partition_num_elems])
        self.partitioned_swap_buffer = get_accelerator().pin_memory(torch.zeros(aligned_numel,
                                                                                device='cpu',
                                                                                dtype=self.dtype),
                                                                    align_bytes=0)
        self.partitioned_swap_pool = SwapBufferPool([self.partitioned_swap_buffer])

    def swap_out_partitioned_params(self, dst_fp16_params, src_fp32_params):
        assert self.partitioned_swap_buffer is not None, 'partitioned swap buffers for fp16 params not initialized'
        assert self.partitioned_swap_pool is not None, 'partitioned swap pool for fp16 params not initialized'
        assert len(dst_fp16_params) == len(src_fp32_params), \
        f'mismatch in number of fp16 params {len(dst_fp16_params)} and fp32 params {len(src_fp32_params)}'

        self.invalidate_cache(dst_fp16_params)

        fp16_swap_paths = self._get_swap_paths(dst_fp16_params, must_exist=True)
        self.synchronize_writes()
        self.partitioned_swap_pool.reset()
        for i, fp32_tensor in enumerate(src_fp32_params):
            swap_tensor, _ = self.partitioned_swap_pool.insert_tensor(fp32_tensor, fp16_swap_paths[i],
                                                                      self._io_aligned_numel(fp32_tensor.numel()))
            assert swap_tensor is not None
            dst_fp16_params[i].ds_tensor.status = PartitionedParamStatus.AVAILABLE

        self.partitioned_swap_pool.swap_out(self.aio_write_handle)

        for param in dst_fp16_params:
            param.ds_tensor.status = PartitionedParamStatus.NOT_AVAILABLE
