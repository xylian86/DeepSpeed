# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import torch

from deepspeed.runtime.zero.superrl_sync import apply_superrl_sync_module_config, select_superrl_sync_leaf_modules
from deepspeed.utils import set_z3_leaf_module, z3_leaf_module


class TinyExpert(torch.nn.Module):

    def __init__(self, hidden_dim=4):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class StructuralMoeBlock(torch.nn.Module):

    def __init__(self, hidden_dim=4):
        super().__init__()
        self.router = torch.nn.Linear(hidden_dim, 2, bias=False)
        self.experts = torch.nn.ModuleList([TinyExpert(hidden_dim), TinyExpert(hidden_dim)])

    def forward(self, x):
        return self.experts[0](x)


class NestedMoeLayer(torch.nn.Module):

    def __init__(self, hidden_dim=4):
        super().__init__()
        self.gate = torch.nn.Linear(hidden_dim, 2, bias=False)
        self.experts = torch.nn.ModuleList([TinyExpert(hidden_dim), TinyExpert(hidden_dim)])

    def forward(self, x):
        return self.experts[0](x)


class NestedMoeBlock(torch.nn.Module):

    def __init__(self, hidden_dim=4):
        super().__init__()
        self.router = torch.nn.Linear(hidden_dim, 2, bias=False)
        self.experts = NestedMoeLayer(hidden_dim)

    def forward(self, x):
        return self.experts(x)


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(4, 4, bias=False)
        self.moe = StructuralMoeBlock()

    def forward(self, x):
        return self.moe(self.dense(x))


def test_superrl_sync_disabled_does_not_select_moe_leaves():
    model = ToyModel()

    selections = apply_superrl_sync_module_config(model, SimpleNamespace(enabled=False))

    assert selections == []
    assert not z3_leaf_module(model.moe)


def test_superrl_sync_marks_structural_moe_block_as_leaf():
    model = ToyModel()

    selections = apply_superrl_sync_module_config(model, SimpleNamespace(enabled=True))

    assert [selection.name for selection in selections] == ["moe"]
    assert z3_leaf_module(model.moe)
    assert not z3_leaf_module(model.moe.experts[0])
    assert selections[0].parameter_count == len(list(model.moe.parameters()))
    assert selections[0].parameter_numel == sum(p.numel() for p in model.moe.parameters())


def test_superrl_sync_selects_outermost_moe_candidate():
    model = torch.nn.Sequential(NestedMoeBlock())

    selections = apply_superrl_sync_module_config(model, SimpleNamespace(enabled=True))

    assert [selection.name for selection in selections] == ["0"]
    assert z3_leaf_module(model[0])
    assert not z3_leaf_module(model[0].experts)


def test_superrl_sync_skips_candidates_covered_by_existing_leaf():
    model = ToyModel()
    set_z3_leaf_module(model, True)

    selections = select_superrl_sync_leaf_modules(model)

    assert selections == []
    assert z3_leaf_module(model)
