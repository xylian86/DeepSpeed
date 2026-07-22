# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import os

from deepspeed import comm as dist
import torch

from deepspeed.runtime.swap_tensor.constants import *
from deepspeed.runtime.swap_tensor.utils import swap_in_tensors, swap_out_tensors, print_object, print_rank_0, \
    SwapBufferManager, get_swap_io_handle_factory
from deepspeed.runtime.swap_tensor.async_swapper import AsyncTensorSwapper
from deepspeed.runtime.swap_tensor.optimizer_utils import OptimizerSwapper
from deepspeed.accelerator import get_accelerator


class OptimizerSwapOp(object):

    def __init__(self, aio_handle, read_op, param_info, allocated_buffers, state_buffers, num_ops, buffer_leases=None):
        self.aio_handle = aio_handle
        self.read_op = read_op
        self.param_info = param_info
        self.allocated_buffers = allocated_buffers
        self.state_buffers = state_buffers
        self.buffer_leases = buffer_leases or []
        self.wait_required = True
        self.num_ops = num_ops

    def is_parameter(self, parameter):
        return OptimizerSwapper.parameter_id(parameter) == self.param_info.param_id

    def wait(self):
        assert self.wait_required
        assert self.aio_handle.wait() == self.num_ops
        self.wait_required = False

    def release_buffers(self):
        for lease in self.take_buffer_leases():
            lease.release()

    def take_buffer_leases(self):
        buffer_leases = self.buffer_leases
        self.buffer_leases = []
        return buffer_leases

    def occupancy(self):
        return {
            'param_id': self.param_info.param_id,
            'kind': 'read' if self.read_op else 'write',
            'wait_required': self.wait_required,
            'lease_count': len(self.buffer_leases),
            'lease_buffer_count': sum(len(lease) for lease in self.buffer_leases),
            'allocated_buffer_count': len(self.allocated_buffers),
            'state_buffer_count': len(self.state_buffers),
            'num_ops': self.num_ops,
        }


SYNC_SWAP_IN = 'sync_swap_in'
ASYNC_SWAP_IN = 'async_swap_in'
SYNC_SWAP_OUT = 'sync_swap_out'
ASYNC_SWAP_OUT = 'async_swap_out'

PIPELINE_OCCUPANCY_LOG_ENV = 'DEEPSPEED_NVME_PIPELINE_OCCUPANCY_LOG'

SWAP_IN_STATE_TIMER = 'swap_in_state'
SWAP_OUT_STATE_TIMER = 'swap_out_state'
SWAP_OUT_GRADIENT_TIMER = 'swap_out_gradient'
ASYNC_SWAP_IN_STATE_TIMER = "async_swap_in_state"
ASYNC_SWAP_OUT_STATE_TIMER = 'async_swap_out_state'


class PipelinedOptimizerSwapper(OptimizerSwapper):

    def __init__(self, swap_config, aio_config, base_folder, optimizer, largest_numel, device, dtype, timers):
        super(PipelinedOptimizerSwapper, self).__init__(swap_config, aio_config, base_folder, optimizer, largest_numel,
                                                        device, dtype, timers)

        aio_handle_factory = get_swap_io_handle_factory(aio_config)
        self.write_aio_handle = aio_handle_factory(block_size=aio_config[AIO_BLOCK_SIZE],
                                                   queue_depth=aio_config[AIO_QUEUE_DEPTH],
                                                   single_submit=aio_config[AIO_SINGLE_SUBMIT],
                                                   overlap_events=aio_config[AIO_OVERLAP_EVENTS],
                                                   intra_op_parallelism=aio_config[AIO_INTRA_OP_PARALLELISM])

        self.read_aio_handle = aio_handle_factory(block_size=aio_config[AIO_BLOCK_SIZE],
                                                  queue_depth=aio_config[AIO_QUEUE_DEPTH],
                                                  single_submit=aio_config[AIO_SINGLE_SUBMIT],
                                                  overlap_events=aio_config[AIO_OVERLAP_EVENTS],
                                                  intra_op_parallelism=aio_config[AIO_INTRA_OP_PARALLELISM])
        self.read_aio_handles = [self.read_aio_handle]

        # Overlap gradient swap out
        self.gradient_swapper = AsyncTensorSwapper(aio_handle=self.write_aio_handle,
                                                   numel_alignment=self.numel_alignment,
                                                   timers=self.timers)

        self.async_swap_in = swap_config.pipeline_read
        self.async_swap_out = swap_config.pipeline_write

        self.swap_ops = {SYNC_SWAP_IN: None, ASYNC_SWAP_IN: [], SYNC_SWAP_OUT: None, ASYNC_SWAP_OUT: None}
        self.pipeline_occupancy_events = 0
        self.pipeline_occupancy_log_enabled = os.environ.get(PIPELINE_OCCUPANCY_LOG_ENV, '').lower() in [
            '1', 'true', 'yes', 'on'
        ]

        self.print_exclude_list += [
            'gradient_swapper', 'read_aio_handle', 'write_aio_handle', 'swap_ops', 'pipeline_occupancy_events',
            'pipeline_occupancy_log_enabled', 'print_exclude_list'
        ]

        if dist.get_rank() == 0:
            print_object(obj=self, name='PipelinedOptimizerSwapper', exclude_list=self.print_exclude_list)

    def initialize_parameters(self, parameters, src_tensors):
        self._initialize_parameters(parameters=parameters, src_tensors=src_tensors, aio_handle=self.write_aio_handle)

    def initialize_from_swapped_fp16_params(self, fp16_partitions_info, fp16_num_elems, fp16_pinned_buffers,
                                            fp32_parameters):
        self._initialize_from_swapped_fp16_params(aio_handle=self.write_aio_handle,
                                                  fp16_partitions_info=fp16_partitions_info,
                                                  fp16_num_elems=fp16_num_elems,
                                                  fp16_pinned_buffers=fp16_pinned_buffers,
                                                  fp32_parameters=fp32_parameters)

    def flush_gradients(self):
        self._flush_gradient_swapper(self.gradient_swapper)

    def swap_in_optimizer_state(self, parameter, async_parameter):
        assert parameter is not None
        assert self.swap_ops[SYNC_SWAP_IN] is None

        self._flush_gradient_swapper(self.gradient_swapper)

        self._start_timer(SWAP_IN_STATE_TIMER)

        async_swap_in_queue = self._async_swap_in_queue()
        if async_swap_in_queue:
            assert async_swap_in_queue[0].is_parameter(parameter)
            self.swap_ops[SYNC_SWAP_IN] = async_swap_in_queue.pop(0)
        else:
            self.swap_ops[SYNC_SWAP_IN] = self._swap_in_optimizer_state(
                aio_handle=self._next_available_read_aio_handle(async_swap_in_queue), parameter=parameter)

        if self.swap_ops[SYNC_SWAP_IN]:
            self.swap_ops[SYNC_SWAP_IN].wait()

        if self.async_swap_in:
            queued_param_ids = {op.param_info.param_id for op in async_swap_in_queue}
            current_param_id = OptimizerSwapper.parameter_id(parameter)
            for candidate in self._normalize_async_parameters(async_parameter):
                candidate_id = OptimizerSwapper.parameter_id(candidate)
                if candidate_id == current_param_id or candidate_id in queued_param_ids:
                    continue
                read_aio_handle = self._next_available_read_aio_handle(async_swap_in_queue)
                if read_aio_handle is None:
                    break
                prefetch_op = self._swap_in_optimizer_state(aio_handle=read_aio_handle,
                                                            parameter=candidate,
                                                            allow_buffer_miss=True)
                if prefetch_op is None:
                    break
                async_swap_in_queue.append(prefetch_op)
                queued_param_ids.add(candidate_id)

        self._stop_timer(SWAP_IN_STATE_TIMER)
        self.timer_names.add(SWAP_IN_STATE_TIMER)
        self._log_pipeline_occupancy('after swap_in_optimizer_state')

    def _async_swap_in_queue(self):
        queue = self.swap_ops[ASYNC_SWAP_IN]
        if queue is None:
            queue = []
        elif not isinstance(queue, list):
            queue = [queue]
        self.swap_ops[ASYNC_SWAP_IN] = queue
        return queue

    def _normalize_async_parameters(self, async_parameter):
        if async_parameter is None:
            return []
        if isinstance(async_parameter, (list, tuple)):
            return [parameter for parameter in async_parameter if parameter is not None]
        return [async_parameter]

    def _next_available_read_aio_handle(self, async_swap_in_queue):
        read_aio_handles = getattr(self, 'read_aio_handles', [self.read_aio_handle])
        pending_handle_ids = {id(op.aio_handle) for op in async_swap_in_queue if op.wait_required}
        for read_aio_handle in read_aio_handles:
            if id(read_aio_handle) not in pending_handle_ids:
                return read_aio_handle
        return None

    def swap_out_optimizer_state(self, parameter, async_swap):
        self._start_timer(SWAP_OUT_STATE_TIMER)

        if self.swap_ops[ASYNC_SWAP_OUT]:
            self._log_pipeline_occupancy('before completing previous async swap-out')
            self._start_timer(ASYNC_SWAP_OUT_STATE_TIMER)
            self._complete_swap_out(ASYNC_SWAP_OUT)
            self._stop_timer(ASYNC_SWAP_OUT_STATE_TIMER)
            self.timer_names.add(ASYNC_SWAP_OUT_STATE_TIMER)

        assert self.swap_ops[SYNC_SWAP_IN] is not None
        assert not self.swap_ops[SYNC_SWAP_IN].wait_required
        swap_op = self._swap_out_optimizer_state(aio_handle=self.write_aio_handle,
                                                 parameter=parameter,
                                                 swap_in_op=self.swap_ops[SYNC_SWAP_IN])
        self.swap_ops[SYNC_SWAP_IN] = None

        if self.async_swap_out and async_swap:
            self.swap_ops[ASYNC_SWAP_OUT] = swap_op
        else:
            self.swap_ops[SYNC_SWAP_OUT] = swap_op
            self._complete_swap_out(SYNC_SWAP_OUT)

        self._stop_timer(SWAP_OUT_STATE_TIMER)
        self.timer_names.add(SWAP_OUT_STATE_TIMER)
        self._log_pipeline_occupancy('after swap_out_optimizer_state')

    def swap_out_gradients(self, parameter, gradient_offsets, gradient_tensors):
        self._swap_out_gradients(parameter=parameter,
                                 gradient_offsets=gradient_offsets,
                                 gradient_tensors=gradient_tensors,
                                 gradient_swapper=self.gradient_swapper)

    def _complete_swap_out(self, swap_out_type):
        self._log_pipeline_occupancy(f'before complete {swap_out_type}')
        self.swap_ops[swap_out_type].wait()
        for buffer in self.swap_ops[swap_out_type].state_buffers:
            buffer = torch.Tensor()
        self.swap_ops[swap_out_type].release_buffers()
        self.swap_ops[swap_out_type] = None
        self._log_pipeline_occupancy(f'after complete {swap_out_type}')

    def _pipeline_slot_occupancy(self, swap_op):
        if swap_op is None:
            return None
        if isinstance(swap_op, list):
            return [self._pipeline_slot_occupancy(op) for op in swap_op]
        return swap_op.occupancy()

    def _pipeline_occupancy(self):
        slots = {name: self._pipeline_slot_occupancy(op) for name, op in self.swap_ops.items()}
        active_slots = []
        for slot in slots.values():
            if isinstance(slot, list):
                active_slots += [entry for entry in slot if entry is not None]
            elif slot is not None:
                active_slots.append(slot)
        return {
            'slots': slots,
            'active_slot_count': len(active_slots),
            'lease_count': sum([slot['lease_count'] for slot in active_slots]),
            'lease_buffer_count': sum([slot['lease_buffer_count'] for slot in active_slots]),
            'allocated_buffer_count': sum([slot['allocated_buffer_count'] for slot in active_slots]),
            'state_buffer_count': sum([slot['state_buffer_count'] for slot in active_slots]),
            'pending_io_op_count': sum([slot['num_ops'] for slot in active_slots if slot['wait_required']]),
        }

    def _format_pipeline_slot_occupancy(self, name, slot):
        if slot is None:
            return f'{name}=None'
        if isinstance(slot, list):
            formatted = [self._format_pipeline_slot_occupancy(f'{name}[{index}]', entry)
                         for index, entry in enumerate(slot)]
            return f'{name}=[]' if not formatted else " | ".join(formatted)
        return (
            f"{name}({slot['kind']},param={slot['param_id']},wait={slot['wait_required']},"
            f"leases={slot['lease_count']},lease_buffers={slot['lease_buffer_count']},"
            f"allocated_buffers={slot['allocated_buffer_count']},state_buffers={slot['state_buffer_count']},"
            f"num_ops={slot['num_ops']})")

    def _log_pipeline_occupancy(self, label):
        if not getattr(self, 'pipeline_occupancy_log_enabled', False):
            return

        occupancy = self._pipeline_occupancy()
        slot_summaries = [
            self._format_pipeline_slot_occupancy(name, occupancy['slots'][name]) for name in [
                SYNC_SWAP_IN,
                ASYNC_SWAP_IN,
                SYNC_SWAP_OUT,
                ASYNC_SWAP_OUT,
            ]
        ]
        summary = (
            f"Optimizer pipeline occupancy[{self.pipeline_occupancy_events}] {label}: "
            f"active_slots={occupancy['active_slot_count']}, "
            f"leases={occupancy['lease_count']}, "
            f"lease_buffers={occupancy['lease_buffer_count']}, "
            f"allocated_buffers={occupancy['allocated_buffer_count']}, "
            f"state_buffers={occupancy['state_buffer_count']}, "
            f"pending_io_ops={occupancy['pending_io_op_count']} | " + " | ".join(slot_summaries))
        self.pipeline_occupancy_events += 1
        print_rank_0(summary, force=True)

    def _swap_out_optimizer_state(self, aio_handle, parameter, swap_in_op):
        assert swap_in_op.is_parameter(parameter)

        allocated_buffers = swap_in_op.allocated_buffers.copy()
        buffer_leases = swap_in_op.take_buffer_leases()

        try:
            param_info = swap_in_op.param_info
            self._update_param_state_info(param_info, parameter)
            unbound_direct_tensors = param_info.unbound_direct_tensor_count()
            if unbound_direct_tensors:
                aligned_numel = self._io_aligned_numel(param_info.numel())
                lazy_state_lease = self.swap_buffer_manager.allocate_lease(num_elems=aligned_numel,
                                                                           count=unbound_direct_tensors,
                                                                           dtype=parameter.dtype,
                                                                           owner='pipelined optimizer lazy state')
                if lazy_state_lease is None:
                    raise RuntimeError(
                        self.swap_buffer_manager.allocation_failure_message(
                            requested_num_elems=aligned_numel,
                            requested_count=unbound_direct_tensors,
                            owner='pipelined optimizer lazy state'))
                buffer_leases.append(lazy_state_lease)
                allocated_buffers += lazy_state_lease.buffers
                param_info.bind_unbound_direct_swap_buffers(lazy_state_lease.buffers, aligned_numel)

            unpinned_tensors = param_info.get_unpinned_state_tensors()

            if len(unpinned_tensors) > 0:
                _, unpinned_paths, unpinned_offsets = param_info.get_swap_buffers_and_paths(False)
                staging_lease = self._allocate_staging_lease(owner='pipelined optimizer swap-out staging')
                try:
                    self._swap_out_unpinned_tensors(aio_handle=aio_handle,
                                                    unpinned_tensors=unpinned_tensors,
                                                    dest_paths=unpinned_paths,
                                                    dest_offsets=unpinned_offsets,
                                                    pinned_buffers=staging_lease.buffers)
                finally:
                    staging_lease.release()

            swap_buffers, swap_paths, swap_offsets = param_info.get_swap_buffers_and_paths(True)
            assert len(swap_paths) == len(swap_buffers)

            num_swap_ops = swap_out_tensors(aio_handle, swap_buffers, swap_paths, swap_offsets)

            swap_out_op = OptimizerSwapOp(aio_handle=aio_handle,
                                          param_info=param_info,
                                          read_op=False,
                                          allocated_buffers=allocated_buffers,
                                          state_buffers=swap_buffers,
                                          num_ops=num_swap_ops,
                                          buffer_leases=buffer_leases)
        except Exception:
            for lease in buffer_leases:
                if not lease.released:
                    lease.release()
            raise

        return swap_out_op

    def _swap_in_optimizer_state(self, aio_handle, parameter, allow_buffer_miss=False):
        param_info = self._get_param_swap_info(parameter)
        if param_info is None:
            return None

        num_swap_tensors = param_info.num_tensors()
        required_buffer_count = num_swap_tensors + (1 if param_info.has_gradients() else 0)
        aligned_numel = self._io_aligned_numel(param_info.numel())
        lease = self.swap_buffer_manager.allocate_lease(num_elems=aligned_numel,
                                                        count=required_buffer_count,
                                                        dtype=parameter.dtype,
                                                        owner='pipelined optimizer swap-in')
        if lease is None:
            if allow_buffer_miss:
                return None
            raise RuntimeError(
                self.swap_buffer_manager.allocation_failure_message(
                    requested_num_elems=aligned_numel,
                    requested_count=required_buffer_count,
                    owner='pipelined optimizer swap-in'))
        allocated_buffers = lease.buffers

        try:
            state_buffers = allocated_buffers[:num_swap_tensors]
            param_info.set_swap_buffers(state_buffers, aligned_numel)

            swap_buffers = state_buffers.copy()
            swap_paths = param_info.get_swap_paths()
            swap_offsets = param_info.get_swap_offsets()

            if param_info.has_gradients():
                parameter.grad = allocated_buffers[-1].narrow(0, 0, param_info.numel())
                if param_info.swapped_gradients:
                    swap_buffers += param_info.get_swap_gradient_buffers(parameter.grad)
                    swap_paths += param_info.get_swap_gradient_paths()
                    swap_offsets += param_info.get_swap_gradient_offsets()

            num_swap_ops = swap_in_tensors(aio_handle, swap_buffers, swap_paths, swap_offsets)

            if param_info.unswapped_gradients:
                self._retrieve_unswapped_grad_partitions(swap_info=param_info, dest_buffer=parameter.grad)

            swap_in_op = OptimizerSwapOp(aio_handle=aio_handle,
                                         param_info=param_info,
                                         read_op=True,
                                         allocated_buffers=allocated_buffers,
                                         state_buffers=state_buffers,
                                         num_ops=num_swap_ops,
                                         buffer_leases=[lease])
        except Exception:
            lease.release()
            raise

        return swap_in_op


class SuperRLPipelinedGDSOptimizerSwapper(PipelinedOptimizerSwapper):

    def __init__(self, swap_config, aio_config, base_folder, optimizer, largest_numel, device, dtype, timers):
        read_parallelism = self._gds_parallelism(swap_config, aio_config, 'gds_read_intra_op_parallelism')
        write_parallelism = self._gds_parallelism(swap_config, aio_config, 'gds_write_intra_op_parallelism')
        prefetch_depth = max(1, int(getattr(swap_config, 'gds_prefetch_depth', 1)))

        gds_op = GDSBuilder().load(verbose=False)
        # cuFile driver configuration is process-global and initialized by the
        # first GDS handle. Keep that global cuFile setting write-friendly;
        # the read handle below still owns its larger worker pool.
        self.write_aio_handle = gds_op.gds_handle(block_size=aio_config[AIO_BLOCK_SIZE],
                                                  queue_depth=aio_config[AIO_QUEUE_DEPTH],
                                                  single_submit=aio_config[AIO_SINGLE_SUBMIT],
                                                  overlap_events=aio_config[AIO_OVERLAP_EVENTS],
                                                  intra_op_parallelism=write_parallelism)

        self.read_aio_handle = gds_op.gds_handle(block_size=aio_config[AIO_BLOCK_SIZE],
                                                 queue_depth=aio_config[AIO_QUEUE_DEPTH],
                                                 single_submit=aio_config[AIO_SINGLE_SUBMIT],
                                                 overlap_events=aio_config[AIO_OVERLAP_EVENTS],
                                                 intra_op_parallelism=read_parallelism)
        self.read_aio_handles = [self.read_aio_handle]
        for _ in range(1, prefetch_depth):
            self.read_aio_handles.append(
                gds_op.gds_handle(block_size=aio_config[AIO_BLOCK_SIZE],
                                  queue_depth=aio_config[AIO_QUEUE_DEPTH],
                                  single_submit=aio_config[AIO_SINGLE_SUBMIT],
                                  overlap_events=aio_config[AIO_OVERLAP_EVENTS],
                                  intra_op_parallelism=read_parallelism))

        aio_op = AsyncIOBuilder().load(verbose=False)
        self.init_write_aio_handle = aio_op.aio_handle(block_size=aio_config[AIO_BLOCK_SIZE],
                                                       queue_depth=aio_config[AIO_QUEUE_DEPTH],
                                                       single_submit=aio_config[AIO_SINGLE_SUBMIT],
                                                       overlap_events=aio_config[AIO_OVERLAP_EVENTS],
                                                       intra_op_parallelism=write_parallelism)

        OptimizerSwapper.__init__(self,
                                  swap_config=swap_config,
                                  aio_config=aio_config,
                                  base_folder=base_folder,
                                  optimizer=optimizer,
                                  largest_numel=largest_numel,
                                  device=device,
                                  dtype=dtype,
                                  timers=timers,
                                  buffer_device=get_accelerator().current_device_name(),
                                  pin_memory_fn=self._pin_device_buffer,
                                  unpin_memory_fn=self._unpin_device_buffer,
                                  lazy_swap_buffers=False)

        # Initialization often starts from CPU fp16 partitions. Keep that path
        # on host AIO staging; runtime optimizer state I/O below still uses GDS.
        self.init_staging_swap_buffer_manager = SwapBufferManager(num_elems=self.staging_swap_buffer_manager.num_elems,
                                                                  count=self.staging_swap_buffer_manager.count,
                                                                  dtype=dtype,
                                                                  name='optimizer_init_staging',
                                                                  device='cpu')

        self.gradient_swapper = AsyncTensorSwapper(aio_handle=self.write_aio_handle,
                                                   numel_alignment=self.numel_alignment,
                                                   timers=self.timers)

        self.async_swap_in = swap_config.pipeline_read
        self.async_swap_out = swap_config.pipeline_write

        self.swap_ops = {SYNC_SWAP_IN: None, ASYNC_SWAP_IN: [], SYNC_SWAP_OUT: None, ASYNC_SWAP_OUT: None}
        self.pipeline_occupancy_events = 0
        self.pipeline_occupancy_log_enabled = os.environ.get(PIPELINE_OCCUPANCY_LOG_ENV, '').lower() in [
            '1', 'true', 'yes', 'on'
        ]

        self.print_exclude_list += [
            'gradient_swapper', 'read_aio_handle', 'write_aio_handle', 'init_write_aio_handle',
            'init_staging_swap_buffer_manager', 'swap_ops', 'pipeline_occupancy_events',
            'pipeline_occupancy_log_enabled', 'print_exclude_list'
        ]

        if dist.get_rank() == 0:
            print_object(obj=self, name='SuperRLPipelinedGDSOptimizerSwapper', exclude_list=self.print_exclude_list)

    def _with_init_staging(self, callback):
        original_staging_manager = self.staging_swap_buffer_manager
        self.staging_swap_buffer_manager = self.init_staging_swap_buffer_manager
        try:
            return callback()
        finally:
            self.staging_swap_buffer_manager = original_staging_manager

    def initialize_parameters(self, parameters, src_tensors):
        return self._with_init_staging(lambda: self._initialize_parameters(parameters=parameters,
                                                                          src_tensors=src_tensors,
                                                                          aio_handle=self.init_write_aio_handle))

    def initialize_from_swapped_fp16_params(self, fp16_partitions_info, fp16_num_elems, fp16_pinned_buffers,
                                            fp32_parameters):
        return self._with_init_staging(
            lambda: self._initialize_from_swapped_fp16_params(aio_handle=self.init_write_aio_handle,
                                                             fp16_partitions_info=fp16_partitions_info,
                                                             fp16_num_elems=fp16_num_elems,
                                                             fp16_pinned_buffers=fp16_pinned_buffers,
                                                             fp32_parameters=fp32_parameters))

    def _pin_device_buffer(self, buffer):
        self.read_aio_handle.pin_device_tensor(buffer)
        return buffer

    def _unpin_device_buffer(self, buffer):
        self.read_aio_handle.unpin_device_tensor(buffer)

    @staticmethod
    def _gds_parallelism(swap_config, aio_config, attr_name):
        value = getattr(swap_config, attr_name, None)
        if value is None:
            value = aio_config[AIO_INTRA_OP_PARALLELISM]
        return max(1, int(value))
