# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from contextvars import ContextVar
from dataclasses import dataclass
from importlib import import_module
from importlib import metadata as importlib_metadata
import inspect
import types
from typing import Any

from packaging import version
import torch

import deepspeed.comm as dist
from deepspeed.utils.logging import logger

MIN_FLA_CP_VERSION = "0.4.2"
FORWARD_METADATA_KWARG = "_deepspeed_ulysses_sp_metadata"
CURRENT_FORWARD_METADATA = ContextVar("deepspeed_ulysses_sp_forward_metadata", default=None)
_SUPPORTED_MODEL_TYPES = {"qwen3_5", "qwen3_5_text"}


@dataclass
class SPForwardMetadata:
    full_position_ids: torch.LongTensor
    global_cu_seqlens: torch.LongTensor
    global_cu_seqlens_cpu: torch.LongTensor
    cp_contexts: dict[int, Any]

    @property
    def is_packed(self) -> bool:
        return self.global_cu_seqlens.numel() > 2


@dataclass
class _FLACPOps:
    build_cp_context: Any
    causal_conv1d: Any
    chunk_gated_delta_rule: Any


@dataclass
class _ForwardPatch:
    module: torch.nn.Module
    had_instance_forward: bool
    previous_instance_forward: Any

    def restore(self) -> None:
        if self.had_instance_forward:
            self.module.forward = self.previous_instance_forward
        else:
            self.module.__dict__.pop("forward", None)


def _callable_accepts_keyword(fn, keyword: str) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    if keyword in parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _get_installed_fla_versions() -> dict[str, str]:
    versions = {}
    for distribution_name in ("flash-linear-attention", "fla-core"):
        try:
            versions[distribution_name] = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def load_fla_cp_ops() -> _FLACPOps:
    installed_versions = _get_installed_fla_versions()
    parsed_versions = [version.parse(installed) for installed in installed_versions.values()]
    if parsed_versions and all(installed < version.parse(MIN_FLA_CP_VERSION) for installed in parsed_versions):
        found = ", ".join(f"{name}={installed}" for name, installed in installed_versions.items())
        raise ImportError(
            f"DeepSpeed linear attention CP requires flash-linear-attention/fla-core >= {MIN_FLA_CP_VERSION}; "
            f"found {found}.")

    try:
        cp_module = import_module("fla.ops.cp")
        conv_module = import_module("fla.modules.conv")
        gated_delta_module = import_module("fla.ops.gated_delta_rule")
    except ImportError as exc:
        raise ImportError("DeepSpeed Qwen3.5 linear attention CP requires FLA with fla.ops.cp, fla.modules.conv, and "
                          f"fla.ops.gated_delta_rule support (>= {MIN_FLA_CP_VERSION}).") from exc

    build_cp_context = getattr(cp_module, "build_cp_context", None)
    causal_conv1d = getattr(conv_module, "causal_conv1d", None)
    chunk_gated_delta_rule = getattr(gated_delta_module, "chunk_gated_delta_rule", None)
    missing = [
        name for name, symbol in (
            ("fla.ops.cp.build_cp_context", build_cp_context),
            ("fla.modules.conv.causal_conv1d", causal_conv1d),
            ("fla.ops.gated_delta_rule.chunk_gated_delta_rule", chunk_gated_delta_rule),
        ) if symbol is None
    ]
    if missing:
        raise ImportError(f"Installed FLA is missing required context-parallel symbols: {missing}.")
    for name, fn in (
        ("fla.modules.conv.causal_conv1d", causal_conv1d),
        ("fla.ops.gated_delta_rule.chunk_gated_delta_rule", chunk_gated_delta_rule),
    ):
        if not _callable_accepts_keyword(fn, "cp_context"):
            raise ImportError(f"Installed {name} does not accept cp_context.")
    return _FLACPOps(build_cp_context, causal_conv1d, chunk_gated_delta_rule)


def position_ids_to_packed_cu_seqlens(position_ids: torch.LongTensor) -> torch.LongTensor:
    if position_ids.ndim != 2 or position_ids.shape[0] != 1:
        raise RuntimeError("Linear attention CP requires position_ids with shape [1, seq_len].")
    flat_position_ids = position_ids.reshape(-1)
    sequence_starts = (flat_position_ids == 0).nonzero(as_tuple=False).flatten()
    if sequence_starts.numel() == 0 or sequence_starts[0] != 0:
        sequence_starts = torch.cat((sequence_starts.new_zeros(1), sequence_starts))
    sequence_end = sequence_starts.new_tensor([flat_position_ids.numel()])
    return torch.cat((sequence_starts, sequence_end)).to(dtype=torch.long)


def _normalize_position_ids(position_ids: torch.LongTensor) -> torch.LongTensor:
    if position_ids.ndim == 3 and position_ids.shape[0] in (3, 4):
        position_ids = position_ids[0]
    if position_ids.ndim != 2 or position_ids.shape[0] != 1:
        raise RuntimeError(
            "Qwen3.5 linear attention CP requires padding-free micro batches with position_ids shaped [1, seq_len].")
    return position_ids


def _argument_value(fn, args, kwargs, name: str):
    if name in kwargs:
        return kwargs[name]
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    return bound.arguments.get(name)


def _seq_idx_to_cu_seqlens(seq_idx: torch.LongTensor) -> torch.LongTensor:
    if seq_idx.ndim != 2 or seq_idx.shape[0] != 1:
        raise RuntimeError("seq_idx must have shape [1, seq_len] for linear attention CP.")
    flat = seq_idx.reshape(-1)
    starts = torch.cat((flat.new_zeros(1, dtype=torch.bool), flat[1:] != flat[:-1])).nonzero().flatten()
    if starts.numel() == 0 or starts[0] != 0:
        starts = torch.cat((starts.new_zeros(1), starts))
    return torch.cat((starts, starts.new_tensor([flat.numel()]))).to(torch.long)


def _gated_delta_state_layout_kwargs(fn) -> dict[str, bool]:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "state_v_first" in parameters:
        return {"state_v_first": True}
    if "transpose_state_layout" in parameters:
        return {"transpose_state_layout": True}
    return {}


def _apply_attention_mask(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None or attention_mask.ndim > hidden_states.ndim:
        return hidden_states
    mask = attention_mask if attention_mask.dtype == torch.bool else attention_mask > 0
    while mask.ndim < hidden_states.ndim:
        mask = mask.unsqueeze(-1)
    return hidden_states * mask.to(device=hidden_states.device, dtype=hidden_states.dtype)


def _gated_delta_cp_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    metadata: SPForwardMetadata,
    fla_ops: _FLACPOps,
    cache_params=None,
    attention_mask=None,
    **kwargs,
):
    if cache_params is not None:
        raise RuntimeError("Qwen3.5 linear attention CP is training/prefill-only and does not support cache_params.")
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise RuntimeError(
            f"FLA linear attention CP expects hidden_states [1, local_seq_len, hidden_size], got {hidden_states.shape}."
        )

    hidden_states = _apply_attention_mask(hidden_states, attention_mask)
    batch_size, local_seq_len, _ = hidden_states.shape
    cp_context = metadata.cp_contexts[module.conv_kernel_size]

    mixed_qkv = module.in_proj_qkv(hidden_states)
    z_gate = module.in_proj_z(hidden_states)
    beta_logits = module.in_proj_b(hidden_states)
    gate_logits = module.in_proj_a(hidden_states)

    conv_result = fla_ops.causal_conv1d(
        x=mixed_qkv,
        weight=module.conv1d.weight.squeeze(1).contiguous(),
        bias=module.conv1d.bias,
        activation=module.activation,
        cp_context=cp_context,
    )
    mixed_qkv = conv_result[0] if isinstance(conv_result, tuple) else conv_result

    key_dim = module.num_k_heads * module.head_k_dim
    value_dim = module.num_v_heads * module.head_v_dim
    expected_qkv_dim = 2 * key_dim + value_dim
    if mixed_qkv.shape[-1] != expected_qkv_dim:
        raise RuntimeError(
            f"Unexpected Qwen3.5 gated-delta projection dimension {mixed_qkv.shape[-1]}; expected {expected_qkv_dim}.")

    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(batch_size, local_seq_len, module.num_k_heads, module.head_k_dim)
    key = key.reshape(batch_size, local_seq_len, module.num_k_heads, module.head_k_dim)
    value = value.reshape(batch_size, local_seq_len, module.num_v_heads, module.head_v_dim)
    if module.num_v_heads % module.num_k_heads != 0:
        raise RuntimeError("Qwen3.5 num_v_heads must be divisible by num_k_heads.")
    heads_per_key = module.num_v_heads // module.num_k_heads
    if heads_per_key > 1:
        query = query.repeat_interleave(heads_per_key, dim=2)
        key = key.repeat_interleave(heads_per_key, dim=2)

    beta = beta_logits.sigmoid()
    gate = -module.A_log.float().exp() * torch.nn.functional.softplus(gate_logits.float() + module.dt_bias)
    core_attn_out, _ = fla_ops.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=gate,
        beta=beta,
        cp_context=cp_context,
        use_qk_l2norm_in_kernel=True,
        **_gated_delta_state_layout_kwargs(fla_ops.chunk_gated_delta_rule),
    )

    core_attn_out = core_attn_out.reshape(-1, module.head_v_dim)
    z_gate = z_gate.reshape(-1, module.head_v_dim)
    core_attn_out = module.norm(core_attn_out, z_gate)
    core_attn_out = core_attn_out.reshape(batch_size, local_seq_len, value_dim)
    return module.out_proj(core_attn_out)


class Qwen35LinearAttentionCPRegistration:

    def __init__(
        self,
        model: torch.nn.Module,
        text_model: torch.nn.Module,
        decoder_layers: list[torch.nn.Module],
        linear_layers: list[torch.nn.Module],
        fla_ops: _FLACPOps,
        sp_group,
        sp_world_size: int,
        core_attn_implementation: str,
        disable_in_eval: bool,
    ):
        self.model = model
        self.text_model = text_model
        self.decoder_layers = decoder_layers
        self.linear_layers = linear_layers
        self.fla_ops = fla_ops
        self.sp_group = sp_group
        self.sp_world_size = sp_world_size
        self.core_attn_implementation = core_attn_implementation
        self.disable_in_eval = disable_in_eval
        self._patches: list[_ForwardPatch] = []
        self._installed = False

    def _patch_forward(self, module: torch.nn.Module, forward_fn) -> None:
        self._patches.append(_ForwardPatch(module, "forward" in module.__dict__, module.__dict__.get("forward")))
        module.forward = types.MethodType(forward_fn, module)

    def _build_metadata(self, position_ids, explicit_cu_seqlens=None, seq_idx=None) -> SPForwardMetadata:
        local_position_ids = _normalize_position_ids(position_ids).contiguous()
        shards = [torch.empty_like(local_position_ids) for _ in range(self.sp_world_size)]
        dist.all_gather(shards, local_position_ids, group=self.sp_group)
        full_position_ids = torch.cat(shards, dim=-1)

        if explicit_cu_seqlens is not None:
            global_cu_seqlens = explicit_cu_seqlens.to(device=full_position_ids.device, dtype=torch.long)
        elif seq_idx is not None:
            local_seq_idx = _normalize_position_ids(seq_idx).contiguous()
            seq_idx_shards = [torch.empty_like(local_seq_idx) for _ in range(self.sp_world_size)]
            dist.all_gather(seq_idx_shards, local_seq_idx, group=self.sp_group)
            global_cu_seqlens = _seq_idx_to_cu_seqlens(torch.cat(seq_idx_shards, dim=-1))
        else:
            global_cu_seqlens = position_ids_to_packed_cu_seqlens(full_position_ids)

        if global_cu_seqlens.ndim != 1 or global_cu_seqlens[-1] != full_position_ids.shape[-1]:
            raise RuntimeError(
                "Packed metadata must contain global cumulative sequence lengths ending at the global sequence length."
            )
        if self.core_attn_implementation == "sdpa" and global_cu_seqlens.numel() > 2:
            raise RuntimeError(
                "Packed Ulysses SP is not supported with SDPA. Use flash_attention_2/3 or flex_attention.")

        global_cu_seqlens_cpu = global_cu_seqlens.detach().cpu()
        cp_contexts = {
            kernel_size:
            self.fla_ops.build_cp_context(
                cu_seqlens=global_cu_seqlens,
                cu_seqlens_cpu=global_cu_seqlens_cpu,
                group=self.sp_group,
                conv1d_kernel_size=kernel_size,
            )
            for kernel_size in {layer.conv_kernel_size
                                for layer in self.linear_layers}
        }
        return SPForwardMetadata(full_position_ids, global_cu_seqlens, global_cu_seqlens_cpu, cp_contexts)

    def install(self) -> None:
        if self._installed:
            return

        original_text_forward = self.text_model.forward
        registration = self

        def text_forward(module, *args, **kwargs):
            if registration.disable_in_eval and not module.training:
                return original_text_forward(*args, **kwargs)
            position_ids = _argument_value(original_text_forward, args, kwargs, "position_ids")
            if position_ids is None:
                raise RuntimeError(
                    "Qwen3.5 Ulysses SP requires position_ids in the input batch before sequence sharding.")
            metadata = registration._build_metadata(
                position_ids,
                explicit_cu_seqlens=kwargs.get("cu_seq_lens_q"),
                seq_idx=kwargs.get("seq_idx"),
            )
            call_kwargs = dict(kwargs)
            call_kwargs[FORWARD_METADATA_KWARG] = metadata
            token = CURRENT_FORWARD_METADATA.set(metadata)
            try:
                return original_text_forward(*args, **call_kwargs)
            finally:
                CURRENT_FORWARD_METADATA.reset(token)

        self._patch_forward(self.text_model, text_forward)

        for decoder_layer in self.decoder_layers:
            original_decoder_forward = decoder_layer.forward
            is_linear = getattr(decoder_layer, "block_type", None) == "linear_attention"

            def make_decoder_forward(original_forward, linear_layer):

                def decoder_forward(module, *args, **kwargs):
                    metadata = kwargs.pop(FORWARD_METADATA_KWARG, None)
                    if metadata is not None and linear_layer:
                        kwargs[FORWARD_METADATA_KWARG] = metadata
                    token = CURRENT_FORWARD_METADATA.set(metadata) if metadata is not None else None
                    try:
                        return original_forward(*args, **kwargs)
                    finally:
                        if token is not None:
                            CURRENT_FORWARD_METADATA.reset(token)

                return decoder_forward

            self._patch_forward(decoder_layer, make_decoder_forward(original_decoder_forward, is_linear))

        for linear_layer in self.linear_layers:
            original_linear_forward = linear_layer.forward

            def make_linear_forward(original_forward):

                def linear_forward(module, hidden_states, cache_params=None, attention_mask=None, *args, **kwargs):
                    metadata = kwargs.pop(FORWARD_METADATA_KWARG, None) or CURRENT_FORWARD_METADATA.get()
                    if metadata is None or (registration.disable_in_eval and not module.training):
                        return original_forward(
                            hidden_states,
                            *args,
                            cache_params=cache_params,
                            attention_mask=attention_mask,
                            **kwargs,
                        )
                    return _gated_delta_cp_forward(
                        module,
                        hidden_states,
                        metadata,
                        registration.fla_ops,
                        cache_params=cache_params,
                        attention_mask=attention_mask,
                        **kwargs,
                    )

                try:
                    from transformers.integrations.accelerate import force_accelerate_hooks
                    return force_accelerate_hooks("conv1d")(linear_forward)
                except ImportError:
                    return linear_forward

            self._patch_forward(linear_layer, make_linear_forward(original_linear_forward))

        self._installed = True
        logger.info(f"[ulysses_sp] installed Qwen3.5 linear-attention CP on {len(self.linear_layers)} model instances")

    def restore(self) -> None:
        for patch in reversed(self._patches):
            patch.restore()
        self._patches.clear()
        self._installed = False


def _config_uses_linear_attention(hf_model_config, arch_cfg) -> bool:
    for config in (hf_model_config, arch_cfg):
        layer_types = getattr(config, "layer_types", None) or ()
        if any(str(layer_type).lower() == "linear_attention" for layer_type in layer_types):
            return True
    return False


def prepare_qwen35_linear_attention_cp(
    model,
    hf_model_config,
    arch_cfg,
    core_attn_implementation: str,
    disable_in_eval: bool,
):
    if not _config_uses_linear_attention(hf_model_config, arch_cfg):
        return None

    model_types = {str(getattr(config, "model_type", "") or "").lower() for config in (hf_model_config, arch_cfg)}
    if not model_types.intersection(_SUPPORTED_MODEL_TYPES):
        raise RuntimeError(
            f"Ulysses SP found linear_attention layers for model types {sorted(model_types)}, but only Qwen3.5 "
            "has a validated FLA context-parallel adapter.")
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(
            "Qwen3.5 linear attention CP requires an instantiated model so DeepSpeed can install model-scoped, "
            "reversible adapters. Pass the model object rather than a model name/path.")

    candidates = []
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and any(hasattr(layer, "linear_attn") for layer in layers):
            candidates.append(module)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one Qwen3.5 text backbone with linear attention layers, found {len(candidates)}.")

    text_model = candidates[0]
    decoder_layers = list(text_model.layers)
    linear_layers = [layer.linear_attn for layer in decoder_layers if hasattr(layer, "linear_attn")]
    required_attrs = (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "conv1d",
        "activation",
        "num_v_heads",
        "num_k_heads",
        "head_k_dim",
        "head_v_dim",
        "conv_kernel_size",
        "A_log",
        "dt_bias",
        "norm",
        "out_proj",
    )
    for layer in linear_layers:
        missing = [attribute for attribute in required_attrs if not hasattr(layer, attribute)]
        if missing:
            raise RuntimeError(
                f"Unsupported {type(layer).__module__}.{type(layer).__name__}; missing attributes {missing}.")

    fla_ops = load_fla_cp_ops()
    return {
        "model": model,
        "text_model": text_model,
        "decoder_layers": decoder_layers,
        "linear_layers": linear_layers,
        "fla_ops": fla_ops,
        "core_attn_implementation": core_attn_implementation,
        "disable_in_eval": disable_in_eval,
    }


def create_qwen35_linear_attention_registration(prepared, sp_group, sp_world_size):
    if prepared is None:
        return None
    return Qwen35LinearAttentionCPRegistration(
        sp_group=sp_group,
        sp_world_size=sp_world_size,
        **prepared,
    )
