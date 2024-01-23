# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator

from utils import DistributedCompileTest


@pytest.fixture
def base_config():
    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
        "fp16": {
            "enabled": True
        },
        "compile": {
            "enabled": True,
            "backend": "inductor"
        }
    }
    return config_dict


class SmallModelWithCustomMethod(torch.nn.Module):

    def __init__(self, hidden_dim, test_value):
        super(SmallModelWithCustomMethod, self).__init__()
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = test_value

    def forward(self, x):
        return self.fc(x)

    # Custom function that is not part of DeepSpeed engine.
    def get_v(self):
        return self.v


class TestCustomMethod(DistributedCompileTest):
    world_size = 1

    def _init_engine(self, config, test_value):
        hidden_dim = 10
        model = SmallModelWithCustomMethod(hidden_dim, test_value)
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        return engine

    def _run_model(self, engine):
        train_batch_size = 1
        device = torch.device(get_accelerator().current_device_name())
        dtype = engine.module.fc.weight.dtype
        hidden_dim = engine.module.fc.weight.shape[1]
        x = torch.rand(train_batch_size, hidden_dim, device=device, dtype=dtype)
        engine(x)

    @pytest.mark.skipif(not deepspeed.compiler.is_compile_supported(), reason="torch.compile is not supported")
    def test_custom_function(self, base_config):
        test_value = 10

        engine = self._init_engine(base_config, test_value)
        assert engine.module.get_v() == test_value
        self._run_model(engine)

        # The model is compiled after the first run.
        # Thus we make sure the custom method is still available after compilation.
        assert engine.module.get_v() == test_value