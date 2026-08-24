# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed.comm as dist
from deepspeed.runtime.sequence_parallel.ulysses_linear_attention import (
    create_qwen35_linear_attention_registration,
    prepare_qwen35_linear_attention_cp,
)
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
from unit.common import DistributedTest
from unit.util import torch_assert_close, torch_assert_equal


def _qwen35_classes():
    pytest.importorskip("fla")
    configuration = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    return configuration.Qwen3_5TextConfig, modeling.Qwen3_5ForCausalLM


def _make_model(device):
    Qwen3_5TextConfig, Qwen3_5ForCausalLM = _qwen35_classes()
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=256,
        use_cache=False,
        pad_token_id=0,
        eos_token_id=1,
    )
    config._attn_implementation = "flash_attention_2"
    torch.manual_seed(1234)
    return Qwen3_5ForCausalLM(config).to(device=device, dtype=torch.bfloat16).train()


def _selected_grads(model):
    return (
        model.model.layers[0].linear_attn.in_proj_qkv.weight.grad.detach().float().clone(),
        model.model.layers[-1].self_attn.q_proj.weight.grad.detach().float().clone(),
        model.lm_head.weight.grad.detach().float().clone(),
    )


class TestUlyssesSPQwen35LinearAttention(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("packed", [False, True])
    def test_output_and_gradient_equivalence(self, packed):
        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        world_size = dist.get_world_size()
        singleton_groups = [dist.new_group([group_rank]) for group_rank in range(world_size)]
        singleton_group = singleton_groups[rank]

        generator = torch.Generator(device="cpu").manual_seed(9876 + int(packed))
        input_ids = torch.randint(2, 128, (1, 128), generator=generator)
        position_ids = (torch.cat(
            (torch.arange(80), torch.arange(48))).unsqueeze(0) if packed else torch.arange(128).unsqueeze(0))

        baseline = _make_model(device)
        prepared = prepare_qwen35_linear_attention_cp(
            baseline,
            baseline.config,
            baseline.config.get_text_config(),
            "flash_attention_2",
            False,
        )
        baseline_registration = create_qwen35_linear_attention_registration(prepared, singleton_group, 1)
        baseline_registration.install()
        baseline_logits = baseline(
            input_ids=input_ids.to(device),
            position_ids=position_ids.to(device),
            use_cache=False,
        ).logits
        baseline_logits.float().square().mean().backward()
        baseline_grads = _selected_grads(baseline)
        baseline_registration.restore()

        candidate = _make_model(device)
        UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=candidate,
            core_attn_implementation="flash_attention_2",
            sequence_parallel_size=world_size,
            micro_batch_size=1,
            seq_length=128,
            seq_length_is_variable=False,
        )
        local_input_ids = input_ids.chunk(world_size, dim=1)[rank].to(device)
        local_position_ids = position_ids.chunk(world_size, dim=1)[rank].to(device)
        candidate_logits = candidate(
            input_ids=local_input_ids,
            position_ids=local_position_ids,
            use_cache=False,
        ).logits
        (candidate_logits.float().square().sum() / baseline_logits.numel()).backward()
        candidate_grads = _selected_grads(candidate)
        for gradient in candidate_grads:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)

        expected_logits = baseline_logits.chunk(world_size, dim=1)[rank]
        torch_assert_equal(expected_logits, candidate_logits)
        for expected_gradient, candidate_gradient in zip(baseline_grads, candidate_grads):
            torch_assert_close(expected_gradient, candidate_gradient, atol=1e-5, rtol=1e-5)

        UlyssesSPAttentionHF.unregister_from_transformers("flash_attention_2")

    def test_disable_in_eval_uses_unsharded_original_forward(self):
        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        generator = torch.Generator(device="cpu").manual_seed(4567)
        input_ids = torch.randint(2, 128, (1, 32), generator=generator).to(device)
        position_ids = torch.arange(32, device=device).unsqueeze(0)
        baseline = _make_model(device).eval()
        candidate = _make_model(device).eval()

        with torch.no_grad():
            expected = baseline(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits
        UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=candidate,
            core_attn_implementation="flash_attention_2",
            sequence_parallel_size=self.world_size,
            micro_batch_size=1,
            seq_length_is_variable=True,
            disable_in_eval=True,
        )
        with torch.no_grad():
            actual = candidate(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits

        torch_assert_equal(expected, actual)
        UlyssesSPAttentionHF.unregister_from_transformers("flash_attention_2")
