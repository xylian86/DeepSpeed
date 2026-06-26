# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import os

from deepspeed.ops.op_builder import AsyncIOBuilder
from deepspeed.ops.op_builder import GDSBuilder
from deepspeed import comm as dist
import torch

from deepspeed.runtime.swap_tensor.constants import *
from deepspeed.runtime.swap_tensor.utils import swap_in_tensors, swap_out_tensors, print_object, print_rank_0
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

        aio_op = AsyncIOBuilder().load()
        self.write_aio_handle = aio_op.aio_handle(block_size=aio_config[AIO_BLOCK_SIZE],
                                                  queue_depth=aio_config[AIO_QUEUE_DEPTH],
                                                  single_submit=aio_config[AIO_SINGLE_SUBMIT],
                                                  overlap_events=aio_config[AIO_OVERLAP_EVENTS],
                                                  intra_op_parallelism=aio_config[AIO_INTRA_OP_PARALLELISM])

        self.read_aio_handle = aio_op.aio_handle(block_size=aio_config[AIO_BLOCK_SIZE],
                                                 queue_depth=aio_config[AIO_QUEUE_DEPTH],
                                                 single_submit=aio_config[AIO_SINGLE_SUBMIT],
                                                 overlap_events=aio_config[AIO_OVERLAP_EVENTS],
                                                 intra_op_parallelism=aio_config[AIO_INTRA_OP_PARALLELISM])

        # Overlap gradient swap out
        self.gradient_swapper = AsyncTensorSwapper(aio_handle=self.write_aio_handle,
                                                   numel_alignment=self.numel_alignment,
                                                   timers=self.timers)

        self.async_swap_in = swap_config.pipeline_read
        self.async_swap_out = swap_config.pipeline_write

        self.swap_ops = {SYNC_SWAP_IN: None, ASYNC_SWAP_IN: None, SYNC_SWAP_OUT: None, ASYNC_SWAP_OUT: None}
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

        if self.swap_ops[ASYNC_SWAP_IN]:
            assert self.swap_ops[ASYNC_SWAP_IN].is_parameter(parameter)
            self.swap_ops[SYNC_SWAP_IN] = self.swap_ops[ASYNC_SWAP_IN]
            self.swap_ops[ASYNC_SWAP_IN] = None
        else:
            self.swap_ops[SYNC_SWAP_IN] = self._swap_in_optimizer_state(aio_handle=self.read_aio_handle,
                                                                        parameter=parameter)

        if self.swap_ops[SYNC_SWAP_IN]:
            self.swap_ops[SYNC_SWAP_IN].wait()

        if self.async_swap_in and async_parameter is not None:
            assert self.swap_ops[ASYNC_SWAP_IN] is None
            self.swap_ops[ASYNC_SWAP_IN] = self._swap_in_optimizer_state(aio_handle=self.read_aio_handle,
                                                                         parameter=async_parameter)

        self._stop_timer(SWAP_IN_STATE_TIMER)
        self.timer_names.add(SWAP_IN_STATE_TIMER)
        self._log_pipeline_occupancy('after swap_in_optimizer_state')

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
        return swap_op.occupancy()

    def _pipeline_occupancy(self):
        slots = {name: self._pipeline_slot_occupancy(op) for name, op in self.swap_ops.items()}
        active_slots = [slot for slot in slots.values() if slot is not None]
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

    def _swap_in_optimizer_state(self, aio_handle, parameter):
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
        gds_op = GDSBuilder().load(verbose=False)
        self.write_aio_handle = gds_op.gds_handle(block_size=aio_config[AIO_BLOCK_SIZE],
                                                  queue_depth=aio_config[AIO_QUEUE_DEPTH],
                                                  single_submit=aio_config[AIO_SINGLE_SUBMIT],
                                                  overlap_events=aio_config[AIO_OVERLAP_EVENTS],
                                                  intra_op_parallelism=aio_config[AIO_INTRA_OP_PARALLELISM])

        self.read_aio_handle = gds_op.gds_handle(block_size=aio_config[AIO_BLOCK_SIZE],
                                                 queue_depth=aio_config[AIO_QUEUE_DEPTH],
                                                 single_submit=aio_config[AIO_SINGLE_SUBMIT],
                                                 overlap_events=aio_config[AIO_OVERLAP_EVENTS],
                                                 intra_op_parallelism=aio_config[AIO_INTRA_OP_PARALLELISM])

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
                                  unpin_memory_fn=self._unpin_device_buffer)

        self.gradient_swapper = AsyncTensorSwapper(aio_handle=self.write_aio_handle,
                                                   numel_alignment=self.numel_alignment,
                                                   timers=self.timers)

        self.async_swap_in = True
        self.async_swap_out = True

        self.swap_ops = {SYNC_SWAP_IN: None, ASYNC_SWAP_IN: None, SYNC_SWAP_OUT: None, ASYNC_SWAP_OUT: None}
        self.pipeline_occupancy_events = 0
        self.pipeline_occupancy_log_enabled = os.environ.get(PIPELINE_OCCUPANCY_LOG_ENV, '').lower() in [
            '1', 'true', 'yes', 'on'
        ]

        self.print_exclude_list += [
            'gradient_swapper', 'read_aio_handle', 'write_aio_handle', 'swap_ops', 'pipeline_occupancy_events',
            'pipeline_occupancy_log_enabled', 'print_exclude_list'
        ]

        if dist.get_rank() == 0:
            print_object(obj=self, name='SuperRLPipelinedGDSOptimizerSwapper', exclude_list=self.print_exclude_list)

    def _pin_device_buffer(self, buffer):
        self.read_aio_handle.pin_device_tensor(buffer)
        return buffer

    def _unpin_device_buffer(self, buffer):
        self.read_aio_handle.unpin_device_tensor(buffer)
