# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TraceRecorder + merge_traces."""
import torch
import torch.nn as nn

from deepspeed.runtime.superrl.cache.trace_recorder import (
    TraceRecorder,
    merge_traces,
    translate_param_ids_to_ds_ids,
    _lcs,
    _dedup_keep_first,
)


class _TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def test_records_forward_and_backward_in_param_order():
    """Recorder emits ``id(param)`` ints, ordered by execution, so the
    DeepSpeed engine can later translate to ``ds_id`` for the cache."""
    model = _TwoLayer()
    rec = TraceRecorder(model, moe_leaf_aggregation=True)
    rec.start_step()
    x = torch.randn(2, 8, requires_grad=True)
    out = model(x).sum()
    out.backward()
    rec.end_step()
    rec.detach()

    traces = rec.traces()
    assert len(traces) == 1
    t = traces[0]
    fc1_w, fc1_b = id(model.fc1.weight), id(model.fc1.bias)
    fc2_w, fc2_b = id(model.fc2.weight), id(model.fc2.bias)
    # Forward: fc1 (weight, bias) then fc2 (weight, bias).
    # Backward: fc2 then fc1, same intra-module order.
    assert t == [fc1_w, fc1_b, fc2_w, fc2_b, fc2_w, fc2_b, fc1_w, fc1_b], t
    # Every recorded id should also appear in param_map.
    pmap = rec.param_map()
    for pid in {fc1_w, fc1_b, fc2_w, fc2_b}:
        assert pid in pmap


def test_no_double_counting_on_nested_leaf_keyword():
    """An MLP block (matched by keyword) should NOT also fire on its
    nn.Linear children, otherwise the trace double-counts."""

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(4, 8)
            self.down = nn.Linear(8, 4)

        def forward(self, x):
            return self.down(self.up(x))

    model = MLP()
    rec = TraceRecorder(model, moe_leaf_aggregation=True)
    rec.start_step()
    model(torch.randn(2, 4, requires_grad=True)).sum().backward()
    rec.end_step()
    rec.detach()
    t = rec.traces()[0]
    # MLP fires once forward + once backward = 2 hook firings, each
    # emitting 4 param ids (up.w, up.b, down.w, down.b in that order).
    assert len(t) == 8, t
    expected_one = [id(model.up.weight), id(model.up.bias),
                    id(model.down.weight), id(model.down.bias)]
    assert t[:4] == expected_one
    assert t[4:] == expected_one  # backward emits the same set in the same order


def test_translate_param_ids_uses_ds_id():
    model = _TwoLayer()
    # Mimic DeepSpeed's ds_id assignment.
    for i, p in enumerate(model.parameters()):
        p.ds_id = 100 + i
    trace = [id(model.fc1.weight), id(model.fc2.weight), id(model.fc1.weight)]
    out = translate_param_ids_to_ds_ids(trace, model)
    assert out == [model.fc1.weight.ds_id, model.fc2.weight.ds_id, model.fc1.weight.ds_id]


def test_translate_drops_unmanaged_params():
    model = _TwoLayer()
    # Only mark fc1.weight as DeepSpeed-managed.
    model.fc1.weight.ds_id = 7
    trace = [id(model.fc1.weight), id(model.fc2.weight)]
    out = translate_param_ids_to_ds_ids(trace, model)
    assert out == [7]


def test_multiple_steps_accumulate():
    model = _TwoLayer()
    rec = TraceRecorder(model)
    for _ in range(3):
        rec.start_step()
        out = model(torch.randn(2, 8, requires_grad=True)).sum()
        out.backward()
        rec.end_step()
    rec.detach()
    assert len(rec.traces()) == 3


def test_lcs_basic():
    assert _lcs([1, 2, 3], [1, 2, 3]) == [1, 2, 3]
    assert _lcs([1, 2, 3, 4], [1, 3, 4]) == [1, 3, 4]
    assert _lcs([1, 2], [3, 4]) == []


def test_dedup_keep_first():
    assert _dedup_keep_first([1, 2, 1, 3, 2]) == [1, 2, 3]


def test_merge_traces_returns_stable_order():
    # Three traces that all visit {1,2,3,4} in slightly different orders.
    merged = merge_traces([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 3, 2, 4],
    ])
    # The first trace is always honoured at the head, and every observed
    # leaf appears at least once in the merged ordering.
    assert merged[0] == 1
    assert set(merged) == {1, 2, 3, 4}


def test_merge_traces_single_trace_passthrough():
    assert merge_traces([[1, 2, 3, 4]]) == [1, 2, 3, 4]


def test_merge_traces_handles_empty():
    assert merge_traces([]) == []
    assert merge_traces([[]]) == []
