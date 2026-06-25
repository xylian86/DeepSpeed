# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
SuperRL-Sync support for ZeRO-3.

SuperRL-Sync raises synchronization granularity from routed expert tensors to
stable module boundaries. In ZeRO-3 this maps directly onto leaf modules: when
execution enters a leaf, ZeRO gathers all parameters in that subtree in one
deterministic, coalesced operation and reduces the leaf gradients together after
backward. That prevents rank-divergent expert routing from changing collective
ordering.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch

from deepspeed.utils import logger, set_z3_leaf_module, z3_leaf_module

KNOWN_MOE_MODULES = {
    "deepspeed.moe.layer.MoE",
    "deepspeed.moe.sharded_moe.MOELayer",
    "transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock",
    "transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock",
    "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock",
}

MOE_CLASS_NAME_TOKENS = ("moe", "mixtureofexperts")
ROUTER_ATTR_NAMES = ("gate", "router")
EXPERT_ATTR_NAMES = ("experts", "deepspeed_experts", "mlp_experts")


@dataclass(frozen=True)
class SuperRLSyncLeafSelection:
    name: str
    module: torch.nn.Module
    parameter_count: int
    parameter_numel: int


def _fully_qualified_class_name(module):
    cls = module.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _has_module_attr(module, attr_names):
    for attr_name in attr_names:
        attr = getattr(module, attr_name, None)
        if isinstance(attr, torch.nn.Module):
            return True
    return False


def _has_router_and_experts(module):
    return _has_module_attr(module, ROUTER_ATTR_NAMES) and _has_module_attr(module, EXPERT_ATTR_NAMES)


def _class_name_suggests_moe(module):
    class_name = module.__class__.__name__.replace("_", "").lower()
    return any(token in class_name for token in MOE_CLASS_NAME_TOKENS)


def is_superrl_sync_moe_candidate(module):
    if _fully_qualified_class_name(module) in KNOWN_MOE_MODULES:
        return True

    if _has_router_and_experts(module):
        return True

    return _class_name_suggests_moe(module) and _has_module_attr(module, EXPERT_ATTR_NAMES)


def _is_descendant_module_name(name, ancestor_name):
    if ancestor_name == "":
        return name != ""
    return name.startswith(f"{ancestor_name}.")


def _covered_by_existing_leaf(name, existing_leaf_names):
    return any(_is_descendant_module_name(name, leaf_name) for leaf_name in existing_leaf_names)


def _selection_for_module(name, module):
    parameters = list(module.parameters(recurse=True))
    return SuperRLSyncLeafSelection(name=name,
                                    module=module,
                                    parameter_count=len(parameters),
                                    parameter_numel=sum(p.numel() for p in parameters))


def select_superrl_sync_leaf_modules(model) -> List[SuperRLSyncLeafSelection]:
    existing_leaf_names = [name for name, module in model.named_modules() if z3_leaf_module(module)]
    selected: List[Tuple[str, torch.nn.Module]] = []

    for name, module in model.named_modules():
        if not is_superrl_sync_moe_candidate(module):
            continue
        if _covered_by_existing_leaf(name, existing_leaf_names):
            continue
        if any(_is_descendant_module_name(name, selected_name) for selected_name, _ in selected):
            continue
        selected.append((name, module))

    return [_selection_for_module(name, module) for name, module in selected]


def apply_superrl_sync_module_config(model, superrl_sync_config) -> List[SuperRLSyncLeafSelection]:
    if not getattr(superrl_sync_config, "enabled", False):
        return []

    selections = select_superrl_sync_leaf_modules(model)
    for selection in selections:
        set_z3_leaf_module(selection.module, True)

    if selections:
        summary = ", ".join(
            f"{selection.name or '<root>'}({selection.parameter_count} params, {selection.parameter_numel} numel)"
            for selection in selections)
        logger.info(f"SuperRL-Sync: selected {len(selections)} MoE leaf module(s): {summary}")
    else:
        logger.info("SuperRL-Sync: enabled, but no uncovered MoE modules were detected")

    return selections
