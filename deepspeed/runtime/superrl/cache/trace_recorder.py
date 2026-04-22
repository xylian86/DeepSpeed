# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""Parameter access trace recorder for SuperRL-Cache.

Paper sec. IV.B: "Execution-Ordered Access Trace". RL training exhibits a
largely deterministic parameter access order across iterations. We record
this order during a short warm-up pass so the LookaheadDRAMCache can
prefetch parameters in the order they will actually be used.

Hooks:
- ``register_forward_pre_hook`` on each leaf module: fires *before* the
  forward (which is when the swapper would need the params on-GPU).
- ``register_full_backward_hook`` on each leaf module: fires when grads
  flow through the module on the backward pass (which is when the swapper
  needs the params again to compute grads).

For MoE models, dynamic routing makes per-expert access order
non-deterministic. We aggregate at the leaf module level (the smallest
subtree containing all experts of a single MoE block), so the trace sees
one access per MoE block instead of per-expert (paper sec. IV.D).

To make the leaf abstraction usable downstream, the recorder emits the
*flat list of parameter Python ids that belong to the leaf* every time
the hook fires (in deterministic registration order). The DeepSpeed
engine later translates these to ``ds_id`` (the cache lookup key used by
``partitioned_param_swapper``) by walking ``model.parameters()`` once.

Across ``warmup_iters`` warm-up steps the recorder collects K traces;
``merge_traces`` returns the longest common subsequence (a stable order
robust to MoE routing jitter) which is what we install into the cache.
"""

from typing import Dict, List, Sequence

import torch
import torch.nn as nn


# Module class names whose subtrees should be aggregated as one leaf
# (instead of recursing into their children). This is the MoE-leaf
# treatment from paper sec. IV.D, generalised to any module that owns a
# self-contained compute unit.
_LEAF_KEYWORDS = ("experts", "moe", "mlp", "attention", "attn", "ffn")


class TraceRecorder:
    """Records the forward+backward parameter access order of a model.

    Usage::

        recorder = TraceRecorder(model, moe_leaf_aggregation=True)
        for _ in range(warmup_iters):
            recorder.start_step()
            loss = model(inputs); loss.backward()
            recorder.end_step()
        recorder.detach()
        trace = merge_traces(recorder.traces())
    """

    def __init__(self, model: nn.Module, moe_leaf_aggregation: bool = True):
        self.model = model
        self.moe_leaf_aggregation = moe_leaf_aggregation
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._recording = False
        self._current: List[int] = []
        self._all_traces: List[List[int]] = []
        # leaf_id -> ordered list of param Python ids that belong to it.
        self._leaf_to_params: Dict[int, List[int]] = {}
        # arbitrary param Python id -> nn.Parameter (so callers can resolve
        # ds_id later without holding a model reference here).
        self._param_map: Dict[int, torch.nn.Parameter] = {}
        self._attach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_step(self) -> None:
        self._recording = True
        self._current = []

    def end_step(self) -> None:
        self._recording = False
        if self._current:
            self._all_traces.append(self._current)
        self._current = []

    def traces(self) -> List[List[int]]:
        return [list(t) for t in self._all_traces]

    def param_map(self) -> Dict[int, torch.nn.Parameter]:
        return dict(self._param_map)

    def leaf_to_params(self) -> Dict[int, List[int]]:
        return {k: list(v) for k, v in self._leaf_to_params.items()}

    def detach(self) -> None:
        self._recording = False
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _attach(self) -> None:
        if self.moe_leaf_aggregation:
            self._attach_leaf_hooks()
        else:
            self._attach_param_hooks()

    def _attach_param_hooks(self) -> None:
        """One forward_pre_hook per parameter-owning module - fine-grained.

        Emits ``id(param)`` for every owned param, in registration order,
        on each fire. The trace therefore consists of param Python ids
        from start to end.
        """
        for module in self.model.modules():
            params = [p for p in module.parameters(recurse=False) if p.requires_grad]
            if not params:
                continue
            ids = []
            for p in params:
                self._param_map[id(p)] = p
                ids.append(id(p))

            def make_pre(p_ids: Sequence[int]):
                ids_tuple = tuple(p_ids)

                def pre_hook(_mod, _inp):
                    if self._recording:
                        self._current.extend(ids_tuple)

                return pre_hook

            def make_back(p_ids: Sequence[int]):
                ids_tuple = tuple(p_ids)

                def back_hook(_mod, _gi, _go):
                    if self._recording:
                        self._current.extend(ids_tuple)

                return back_hook

            self._hooks.append(module.register_forward_pre_hook(make_pre(ids)))
            self._hooks.append(module.register_full_backward_hook(make_back(ids)))

    def _attach_leaf_hooks(self) -> None:
        """One pair of hooks per "leaf for trace" - stable under MoE routing.

        We do a top-down DFS from ``self.model``: when a module qualifies
        as a leaf-for-trace we attach hooks on it AND stop recursing into
        its children. This avoids double-counting that would happen if we
        also hooked the true-leaf nn.Linear children inside (e.g.) an
        MLP block.

        On fire, the hook emits the ordered list of param Python ids that
        belong to that subtree.
        """
        attached_modules: List[int] = []  # debug aid; not used at runtime

        def _attach_subtree(module: nn.Module) -> None:
            if self._is_leaf_for_trace(module):
                params = [p for p in module.parameters(recurse=True) if p.requires_grad]
                if not params:
                    return
                leaf_id = id(module)
                param_ids = [id(p) for p in params]
                self._leaf_to_params[leaf_id] = param_ids
                for p in params:
                    self._param_map[id(p)] = p

                ids_tuple = tuple(param_ids)

                def pre_hook(_mod, _inp, _ids=ids_tuple):
                    if self._recording:
                        self._current.extend(_ids)

                def back_hook(_mod, _gi, _go, _ids=ids_tuple):
                    if self._recording:
                        self._current.extend(_ids)

                self._hooks.append(module.register_forward_pre_hook(pre_hook))
                self._hooks.append(module.register_full_backward_hook(back_hook))
                attached_modules.append(leaf_id)
                return  # do NOT descend further

            for child in module.children():
                _attach_subtree(child)

        _attach_subtree(self.model)

    @staticmethod
    def _is_leaf_for_trace(module: nn.Module) -> bool:
        children = list(module.children())
        if not children:
            return True
        cls_name = type(module).__name__.lower()
        return any(kw in cls_name for kw in _LEAF_KEYWORDS)


# ----------------------------------------------------------------------
# Trace merging
# ----------------------------------------------------------------------


def merge_traces(traces: Sequence[Sequence[int]]) -> List[int]:
    """Combine ``traces`` from K warm-up steps into one stable ordering.

    Strategy: pairwise reduction with the longest common subsequence
    (LCS) followed by a stable insertion of any items present in some
    trace but not in the LCS. This restores deterministic order under
    MoE routing jitter while still covering every observed leaf at
    least once.
    """
    if not traces:
        return []
    if len(traces) == 1:
        return _dedup_keep_first(list(traces[0]))

    base = _dedup_keep_first(list(traces[0]))
    for t in traces[1:]:
        ded = _dedup_keep_first(list(t))
        common = _lcs(base, ded)
        if not common:
            base = ded
            continue
        base = _stable_insert(common, base + ded)
    return base


def translate_param_ids_to_ds_ids(
    trace: Sequence[int], model: nn.Module
) -> List[int]:
    """Map a trace of Python ``id(param)`` ints to DeepSpeed ``ds_id`` ints.

    DeepSpeed assigns ``param.ds_id`` to every model parameter once
    Init/partitioning has run; this is the key the partitioned-param
    swapper uses for both swap-in and cache lookup. ``TraceRecorder``
    cannot record ``ds_id`` directly because the ids are not assigned
    until after model wrap.

    Items missing a ``ds_id`` (e.g. non-DeepSpeed-managed params) are
    dropped. Order is preserved.
    """
    id_to_ds: Dict[int, int] = {}
    for p in model.parameters():
        ds_id = getattr(p, "ds_id", None)
        if ds_id is not None:
            id_to_ds[id(p)] = int(ds_id)
    out: List[int] = []
    seen: set = set()
    for pid in trace:
        ds_id = id_to_ds.get(pid)
        if ds_id is None:
            continue
        # Keep duplicates; LookaheadDRAMCache uses positional next-use.
        out.append(ds_id)
        seen.add(ds_id)
    return out


def _dedup_keep_first(seq: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _lcs(a: List[int], b: List[int]) -> List[int]:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    out: List[int] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def _stable_insert(skeleton: List[int], extras: List[int]) -> List[int]:
    """Return ``skeleton`` extended with any item from ``extras`` not yet in it,
    inserted just after its last seen predecessor in ``extras`` to preserve
    locality."""
    out: List[int] = list(skeleton)
    present = set(out)
    for x in extras:
        if x in present:
            continue
        out.append(x)
        present.add(x)
    return out
