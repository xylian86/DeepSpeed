# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping tensors to/from (NVMe) storage devices.
"""

import bisect
import collections
import functools
import math
import os
import shutil
import time
from enum import Enum
import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from .constants import *
from .utils import swap_in_tensors, swap_out_tensors, MIN_AIO_BYTES, AIO_ALIGNED_BYTES, print_object, SwapBufferPool, \
    get_swap_io_handle_factory
from .lookahead_dram_cache import LookaheadDRAMCache
from .superrl_io import apply_superrl_io_gds_env_defaults
from deepspeed.runtime.zero.offload_config import resolve_nvme_path

# With SuperRL-IO the NVMe parameter swap-in reads ahead as far as the swap buffers allow, and a fetch waits only
# for the reads it needs (SUPERRL_PARAM_DEEP_PREFETCH=0 turns this off). The buffer_count x buffer_size memory is
# re-cut into slots of the partition sizes, so it holds several times more partitions in flight
# (SUPERRL_PARAM_SIZED_SLOTS=0 keeps buffer_count slots of buffer_size).
PARAM_DEEP_PREFETCH_ENV = "SUPERRL_PARAM_DEEP_PREFETCH"
PARAM_SIZED_SLOTS_ENV = "SUPERRL_PARAM_SIZED_SLOTS"
# SUPERRL_PARAM_SWAP_STATS=1: each rank prints where parameter swap-ins wait, about every 30 s (at a forward's start).
PARAM_SWAP_STATS_ENV = "SUPERRL_PARAM_SWAP_STATS"
PARAM_KEEP_SWAP_FILES_ENV = "SUPERRL_KEEP_PARAM_SWAP_FILES"


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

        #keep track of async swap in params and buffers, in submission order, and the read ops of each param
        self.inflight_params = []
        self.inflight_swap_in_buffers = []
        self.inflight_ops = []
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

    def _slot(self, buffer_id):
        slots = getattr(self, "buffer_slots", None)
        if slots is None:
            return buffer_id * self.aligned_elements_per_buffer, self.aligned_elements_per_buffer
        return slots[buffer_id]

    def _slot_view(self, buffer_id, numel):
        offset, size = self._slot(buffer_id)
        assert numel <= size, f"{numel} elements do not fit swap buffer {buffer_id} of {size} elements"
        return self.buffers.narrow(0, int(offset), int(numel))

    def buffer_class(self, numel):
        """Slot size a partition of numel elements is read into: the smallest that holds it (None if none does)."""
        sizes = getattr(self, "slot_sizes", None) or [self.aligned_elements_per_buffer]
        index = bisect.bisect_left(sizes, self._io_aligned_numel(numel))
        return sizes[index] if index < len(sizes) else None

    def free_slots_by_class(self):
        return collections.Counter(self._slot(buffer_id)[1] for buffer_id in self.available_buffer_ids)

    def _sized_slot_plan(self):
        """Slots per partition size within the buffer memory: about the same fraction of each size's partitions, at
        least two (or all) of each, with the largest fraction that fits. None if the minimum does not fit."""
        counts = collections.Counter(self._io_aligned_numel(numel) for numel in self.param_id_to_numel.values())
        budget = self.buffers.numel()

        def plan_for(fraction):
            return {size: min(count, max(2, math.ceil(fraction * count))) for size, count in counts.items()}

        def used(plan):
            return sum(size * count for size, count in plan.items())

        if not counts or used(plan_for(0.0)) > budget:
            return None
        if used(plan_for(1.0)) <= budget:
            return plan_for(1.0)
        low, high = 0.0, 1.0
        for _ in range(50):
            middle = (low + high) / 2
            if used(plan_for(middle)) <= budget:
                low = middle
            else:
                high = middle
        return plan_for(low)

    def _plan_sized_slots(self):
        """Re-cut the buffer memory into slots of the model's partition sizes, once, at the first swap-in that finds
        every slot free (all NVMe partitions are known by then). A buffer_size slot holds one partition however
        small, so for most models the same memory now holds several times more partitions in flight."""
        if getattr(self, "sized_slots_planned", True):
            return
        if self.param_id_to_buffer_id or self.reserved_buffer_ids or self.pending_reads:
            return
        self.sized_slots_planned = True
        plan = self._sized_slot_plan()
        if plan is None:
            print_rank_0(f"SuperRL-IO: two slots of each parameter partition size do not fit the "
                         f"{self.param_buffer_count} x {self.aligned_elements_per_buffer} swap buffers; "
                         f"keeping those", force=True)
            return
        for event in self.buffer_release_events.values():
            event.synchronize()
        self.buffer_release_events.clear()
        slots, offset = [], 0
        for size in sorted(plan, reverse=True):
            for _ in range(plan[size]):
                slots.append((offset, size))
                offset += size
        self.buffer_slots = slots
        self.slot_sizes = sorted(plan)
        self.available_buffer_ids = list(range(len(slots)))
        print_rank_0(f"SuperRL-IO: {len(slots)} parameter swap slots in place of {self.param_buffer_count}, "
                     f"{', '.join(f'{plan[size]} x {size}' for size in sorted(plan, reverse=True))} elements",
                     force=True)

    def _slots_short_for(self, params):
        """Slot sizes still missing for params after each takes the smallest free slot that holds it."""
        free = self.free_slots_by_class()
        sizes = sorted(free)
        short = collections.Counter()
        for param in params:
            needed = self._io_aligned_numel(self.param_id_to_numel[param.ds_id])
            size = next((size for size in sizes if size >= needed and free[size] > 0), None)
            if size is None:
                short[self.buffer_class(needed)] += 1
            else:
                free[size] -= 1
        return short

    def _make_room_for(self, params):
        exclude_param_ids = [p.ds_id for p in params]
        short = self._slots_short_for(params)
        if short:
            self.count_stat("room_short", sum(short.values()))
        for size, count in short.items():
            self.release_available_parameters(count, exclude_param_ids=exclude_param_ids, min_slot_size=size)
        short = self._slots_short_for(params)
        if short and self.pending_reads:
            # Partitions still being read hold the rest: finish them so that they can be released too.
            self.synchronize_reads()
            for size, count in short.items():
                self.release_available_parameters(count, exclude_param_ids=exclude_param_ids, min_slot_size=size)
            short = self._slots_short_for(params)
        return short

    def count_stat(self, key, value=1):
        stats = getattr(self, "swap_stats", None)
        if stats is not None:
            stats[key] += value

    def print_swap_stats(self, every_s=30.0):
        stats = getattr(self, "swap_stats", None)
        now = time.perf_counter()
        if stats is None or now - self.swap_stats_since < every_s:
            return
        items = ", ".join(f"{key} {value:.2f}" if isinstance(value, float) else f"{key} {value}"
                          for key, value in sorted(stats.items()))
        print(f"[superrl_param_swap] rank {dist.get_rank()} {now - self.swap_stats_since:.1f} s: {items}", flush=True)
        stats.clear()
        self.swap_stats_since = now

    def release_prefetched_partitions(self):
        """Release the partitions read ahead but not in use, so that no slot carries a partition into a new pass
        or across a parameter update."""
        if self.pending_reads:
            self.synchronize_reads()
        self.count_stat("unused_released", self.release_available_parameters(len(self.available_params)))
        self.print_swap_stats()

    def _configure_aio(self, ds_config):
        self.swap_config = ds_config.zero_config.offload_param
        torch_dtype_string = str(self.dtype).split(".")[1]
        nvme_path = resolve_nvme_path(self.swap_config)
        if nvme_path is None:
            raise ValueError("ZeRO-3 NVMe parameter offload requires nvme_path or nvme_path_per_local_rank.")
        self.swap_folder = os.path.join(nvme_path, 'zero_stage_3', f'{torch_dtype_string}params',
                                        f'rank{dist.get_rank()}')
        # SUPERRL_KEEP_PARAM_SWAP_FILES=1 reuses the files in place (writes do not truncate), so files preallocated on
        # a fast region of the device stay there; every partition is written at init before it is read.
        if os.environ.get(PARAM_KEEP_SWAP_FILES_ENV) != "1":
            shutil.rmtree(self.swap_folder, ignore_errors=True)
        os.makedirs(self.swap_folder, exist_ok=True)

        self.swap_element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.aio_config = ds_config.aio_config

        self.use_gds = self.aio_config[AIO_USE_GDS]
        superrl_io = getattr(getattr(ds_config, "superrl_io_config", None), "enabled", False)
        self.deep_prefetch = superrl_io and os.environ.get(PARAM_DEEP_PREFETCH_ENV, "1") != "0"
        self.sized_slots = superrl_io and os.environ.get(PARAM_SIZED_SLOTS_ENV, "1") != "0"
        self.sized_slots_planned = not self.sized_slots
        self.swap_stats = collections.Counter() if os.environ.get(PARAM_SWAP_STATS_ENV) == "1" else None
        self.swap_stats_since = time.perf_counter()
        if self.use_gds and superrl_io:
            # cuFile reads its config at the process's first GDS handle and the chunk size at its first op. With the
            # parameters on NVMe those are this swapper's, created before the optimizer swapper that sets the
            # SuperRL-IO defaults, so set them here too.
            apply_superrl_io_gds_env_defaults()
        self.aio_handle = get_swap_io_handle_factory(self.aio_config, use_gds=self.use_gds, verbose=False)

        # Read/Write alignment for each thread during Intra-request parallelism
        self.min_aio_bytes = max(MIN_AIO_BYTES, self.aio_config[AIO_BLOCK_SIZE])
        self.aligned_bytes = AIO_ALIGNED_BYTES * self.aio_config[AIO_INTRA_OP_PARALLELISM]
        self.numel_alignment = self.aligned_bytes // self.swap_element_size

        self.elements_per_buffer = self.swap_config.buffer_size
        self.aligned_elements_per_buffer = self._io_aligned_numel(self.elements_per_buffer)
        self.param_buffer_count = self.swap_config.buffer_count

        # Slot (offset, numel) per buffer id in self.buffers; see _plan_sized_slots for the sized layout.
        self.buffer_slots = [(i * self.aligned_elements_per_buffer, self.aligned_elements_per_buffer)
                             for i in range(self.param_buffer_count)]
        self.slot_sizes = [self.aligned_elements_per_buffer]
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
        # The look-ahead cache does its own read-ahead (the coordinator prefetches through it instead).
        self.deep_prefetch = self.deep_prefetch and self.lookahead_cache is None

        buffer_device = get_accelerator().device_name() if self.use_gds else "cpu"
        self.buffers = torch.empty(int(self.aligned_elements_per_buffer * self.param_buffer_count),
                                   dtype=self.dtype,
                                   device=buffer_device,
                                   requires_grad=False)
        if self.use_gds:
            self.aio_read_handle.pin_device_tensor(self.buffers)
        else:
            self.buffers = get_accelerator().pin_memory(self.buffers, make_copy=False)

        # GDS moves data outside CUDA stream order, and Work.wait() on an all-gather only orders streams: a read into a
        # buffer that was just released can overwrite it while the released partition's all-gather still reads it on
        # the GPU. Each released device buffer records an event on the current stream; the next user waits for it.
        self.buffer_release_events = {}

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
                                   pin_memory_fn=functools.partial(get_accelerator().pin_memory, make_copy=False),
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

    def release_available_parameters(self, count, exclude_param_ids=None, min_slot_size=None):
        """Release up to count available partitions not in use; with min_slot_size, only ones in slots that large,
        smallest slots first."""
        if count <= 0:
            return 0
        if exclude_param_ids is None:
            exclude_param_ids = set()
        else:
            exclude_param_ids = set(exclude_param_ids)
        if not hasattr(self, "available_param_objects"):
            self.available_param_objects = {}

        candidates = list(self.available_params)
        if min_slot_size is not None:
            slot_size = {
                param_id: self._slot(self.param_id_to_buffer_id[param_id])[1]
                for param_id in candidates if param_id in self.param_id_to_buffer_id
            }
            candidates = sorted((param_id for param_id in slot_size if slot_size[param_id] >= min_slot_size),
                                key=slot_size.get)

        released = 0
        for param_id in candidates:
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

            buffer_id = self._take_buffer_id(self.param_id_to_numel[param_id])
            print_rank_0(f"param {param.ds_id} is assigned swap in buffer id {buffer_id}  ")
            self.param_id_to_buffer_id[param_id] = buffer_id
            aligned_swap_numel = self._io_aligned_numel(self.param_id_to_numel[param_id])
            swap_buffer = self._slot_view(buffer_id, aligned_swap_numel)

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

    #waits for inflight nvme reads to complete. With params and deep prefetch, only for the reads up to the last of
    #params: the handle completes ops in submission order.
    def synchronize_reads(self, params=None):
        if self.pending_reads == 0:
            return

        count = len(self.inflight_params)
        if params is not None and getattr(self, "deep_prefetch", False) and hasattr(self.aio_read_handle,
                                                                                   "wait_some"):
            wanted = set(param.ds_id for param in params)
            count = 1 + max((i for i, param in enumerate(self.inflight_params) if param.ds_id in wanted), default=-1)
            if count == 0:
                return

        start = time.perf_counter()
        if count == len(self.inflight_params):
            assert self.pending_reads == self.aio_read_handle.wait()
            self.pending_reads = 0
            self.count_stat("read_waits_all")
        else:
            num_ops = sum(self.inflight_ops[:count])
            assert num_ops == self.aio_read_handle.wait_some(num_ops)
            self.pending_reads -= num_ops
            self.count_stat("read_waits_some")
        self.count_stat("read_wait_s", time.perf_counter() - start)

        done_params = self.inflight_params[:count]
        for param, swap_in_buffer in zip(done_params, self.inflight_swap_in_buffers[:count]):
            param_id = param.ds_id
            compute_buffer = swap_in_buffer.narrow(0, 0, self.param_id_to_numel[param_id])
            param.ds_tensor.data = compute_buffer.data
            param.ds_tensor.status = PartitionedParamStatus.AVAILABLE

        if count == len(self.inflight_params):
            self._mark_params_available(self.inflight_params, numel=self.inflight_numel)
            self.inflight_params = []
            self.inflight_swap_in_buffers = []
            self.inflight_ops = []
            self.inflight_numel = 0
        else:
            done_numel = sum(self.param_id_to_numel[param.ds_id] for param in done_params)
            self._mark_params_available(done_params, numel=done_numel)
            del self.inflight_params[:count]
            del self.inflight_swap_in_buffers[:count]
            del self.inflight_ops[:count]
            self.inflight_numel -= done_numel

    #Removes the memory assignment and releases the buffers
    #Should only be executed after swapping out the tensors
    def remove_partition_and_release_buffers(self, params):
        for param in params:
            param_id = param.ds_id

            if param_id in self.param_id_to_buffer_id.keys():

                buffer_id = self.param_id_to_buffer_id[param_id]

                assert buffer_id is not None, "Missing buffer id for releasing"

                self.available_buffer_ids.append(buffer_id)
                if self.use_gds:
                    event = get_accelerator().Event()
                    event.record()
                    self.buffer_release_events[buffer_id] = event
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

    # book keeping function for inflight swap in. num_swap_ops is a count per param, or a total for all of them
    # (counted on the first param, so a wait for any of them waits for all).
    def _update_inflight_swap_in(self, params, swap_in_buffers, inflight_numel, num_swap_ops=None):
        if num_swap_ops is None:
            num_swap_ops = [1] * len(params)
        elif not isinstance(num_swap_ops, list):
            num_swap_ops = [num_swap_ops] + [0] * (len(params) - 1)
        if not hasattr(self, "inflight_ops"):
            self.inflight_ops = []
        self.inflight_params.extend(params)
        self.inflight_swap_in_buffers.extend(swap_in_buffers)
        self.inflight_ops.extend(num_swap_ops)
        self.inflight_numel += inflight_numel

        for param in params:
            param.ds_tensor.status = PartitionedParamStatus.INFLIGHT

        self.pending_reads += sum(num_swap_ops)

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
            self._plan_sized_slots()
            short = self._make_room_for(params)

            if short:
                ids = [p.ds_id for p in params]
                print_rank_0(
                    f'Not enough swap in buffers {len(self.available_buffer_ids)} for {len(swap_in_paths)} params, ids = {ids}, missing slots {dict(short)}',
                    force=True)
                print_rank_0(
                    f'Num inflight: params {len(self.inflight_params)}, buffers {len(self.inflight_swap_in_buffers)}, numel = {self.inflight_numel}',
                    force=True)
                print_rank_0(
                    f'Num available params: count = {len(self.available_params)}, ids = {self.available_params}, numel = {self.available_numel}',
                    force=True)

            assert not short, f"Not enough buffers {len(self.available_buffer_ids)} for swapping {len(swap_in_paths)}"
            compute_buffers, swap_in_buffers = self._allocate_and_return_buffers_for_swap_in(params)
            inflight_numel = sum([t.numel() for t in compute_buffers])
        else:
            inflight_numel = sum([t.numel() for t in swap_in_buffers])

        # One request list per param, so that a later wait can stop after a given param's reads.
        start = time.perf_counter()
        num_swap_ops = [
            swap_in_tensors(self.aio_read_handle, [buffer], [path])
            for buffer, path in zip(swap_in_buffers, swap_in_paths)
        ]
        self.count_stat("issue_s", time.perf_counter() - start)
        kind = "prefetch" if async_op else "demand"
        self.count_stat(f"{kind}_params", len(params))
        self.count_stat(f"{kind}_gib", inflight_numel * self.buffers.element_size() / 2**30)

        self._update_inflight_swap_in(params, swap_in_buffers, inflight_numel, num_swap_ops)

        if not async_op:
            self.synchronize_reads()

    # Enables swapping into buffer that is out the control of swapper. This is always synchronous
    def swap_into_buffer(self, param, dest_buffer):
        assert param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE, f"param {param.ds_id} is already available or inflight"

        require_swap_buffer = not (get_accelerator().is_pinned(dest_buffer)
                                   and self._is_io_aligned(dest_buffer.numel()))

        if require_swap_buffer:
            assert not self._slots_short_for([param]), f"No buffer available to swap param {param.ds_id}."
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
        buffer_id = self._take_buffer_id(numel)
        self.param_id_to_buffer_id[param_id] = buffer_id
        aligned_swap_numel = self._io_aligned_numel(self.param_id_to_numel[param_id])
        swap_buffer = self._slot_view(buffer_id, aligned_swap_numel)

        self.param_id_to_swap_buffer[param_id] = swap_buffer
        compute_buffer = swap_buffer.narrow(0, 0, self.param_id_to_numel[param_id])
        print_rank_0(f"param {param.ds_id} is assigned swap in buffer id {buffer_id}")
        return compute_buffer

    def _take_buffer_id(self, numel):
        # The smallest free slot that holds numel, and of those the one released first: its release event (the GPU
        # work that last read it) is the likeliest to be done already.
        needed = self._io_aligned_numel(numel)
        fits = [(self._slot(buffer_id)[1], index) for index, buffer_id in enumerate(self.available_buffer_ids)
                if self._slot(buffer_id)[1] >= needed]
        assert fits, f"No free swap buffer holds {numel} elements, free slots {dict(self.free_slots_by_class())}"
        buffer_id = self.available_buffer_ids.pop(min(fits)[1])
        event = self.buffer_release_events.pop(buffer_id, None)
        if event is not None:
            if getattr(self, "swap_stats", None) is not None and not event.query():
                start = time.perf_counter()
                event.synchronize()
                self.count_stat("event_wait_s", time.perf_counter() - start)
                self.count_stat("event_blocks")
            else:
                event.synchronize()
        return buffer_id

    def reserve_available_buffers(self):
        for event in self.buffer_release_events.values():
            event.synchronize()
        self.buffer_release_events.clear()
        buffers = []
        for id in self.available_buffer_ids:
            offset, size = self._slot(id)
            buffers.append(self.buffers.narrow(0, int(offset), int(size)))
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
                                                                                dtype=self.dtype))
        self.partitioned_swap_pool = SwapBufferPool([self.partitioned_swap_buffer])

    def swap_out_partitioned_params(self, dst_fp16_params, src_fp32_params):
        assert self.partitioned_swap_buffer is not None, 'partitioned swap buffers for fp16 params not initialized'
        assert self.partitioned_swap_pool is not None, 'partitioned swap pool for fp16 params not initialized'
        assert len(dst_fp16_params) == len(src_fp32_params), \
        f'mismatch in number of fp16 params {len(dst_fp16_params)} and fp32 params {len(src_fp32_params)}'

        self.invalidate_cache(dst_fp16_params)
        if getattr(self, "deep_prefetch", False):
            self.release_prefetched_partitions()

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
