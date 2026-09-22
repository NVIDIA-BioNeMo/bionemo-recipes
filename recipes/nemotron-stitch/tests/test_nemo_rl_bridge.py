# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemotron_stitch.automodel.checkpoint import ProjectorStateDictAdapterMixin
from nemotron_stitch.nemo_rl.policy import (
    require_replicated_projector_policy_config,
    validate_initial_adapter_provenance,
)
from nemotron_stitch.nemo_rl.vllm_worker import (
    _merge_encoder_hf_overrides,
    _select_vllm_executor_backend,
    require_generation_topology,
)


def test_dtensor_refit_inventory_excludes_frozen_extra_parameters():
    pytest.importorskip("nemo_rl", reason="NeMo RL seam tests need the framework installed")
    pytest.importorskip("nemo_automodel", reason="the DTensor policy worker needs AutoModel")
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import dtensor_params_generator

    class _BaseAdapter:
        def __init__(self):
            self.config = SimpleNamespace(hf_checkpoint_prefix="")

        def convert_single_tensor_to_hf(self, name, tensor, **_kwargs):
            return [(name, tensor)]

    class _Adapter(ProjectorStateDictAdapterMixin, _BaseAdapter):
        PROJECTOR_STATE_PREFIXES = ("learned_start", "learned_end")

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Parameter(torch.ones(2, 2))
            self.learned_start = nn.Parameter(torch.ones(2), requires_grad=False)
            self.learned_end = nn.Parameter(torch.ones(2), requires_grad=False)
            self.state_dict_adapter = _Adapter()

    names = {name for name, _ in dtensor_params_generator(_Model(), torch.bfloat16)}
    assert names == {"backbone"}


def test_initial_adapter_provenance_resolves_the_upstream_layout(tmp_path):
    """The gate resolves the same restore_from layouts NeMo RL's loader does."""
    adapter_dir = tmp_path / "LATEST" / "model"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        '{"peft_type": "LORA", "r": 8, "lora_alpha": 16, "extra": "descriptive"}'
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

    # The weights directory or its 'model' subdirectory both resolve; locked
    # values match recursively and descriptive extras are allowed.
    validate_initial_adapter_provenance(adapter_dir, {"peft_type": "LORA", "r": 8, "lora_alpha": 16})
    validate_initial_adapter_provenance(tmp_path / "LATEST", {"peft_type": "LORA", "lora_alpha": 16})


def test_initial_adapter_provenance_fails_closed(tmp_path):
    adapter_dir = tmp_path / "model"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 8}')
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

    with pytest.raises(ValueError, match="must not be empty"):
        validate_initial_adapter_provenance(adapter_dir, {})
    with pytest.raises(ValueError, match="provenance mismatch for provenance.r"):
        validate_initial_adapter_provenance(adapter_dir, {"r": 16})
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        validate_initial_adapter_provenance(tmp_path / "missing", {"r": 8})

    (adapter_dir / "adapter_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        validate_initial_adapter_provenance(adapter_dir, {"r": 8})


def test_policy_and_generation_topology_contracts():
    require_replicated_projector_policy_config({"dtensor_cfg": {"tensor_parallel_size": 1}})
    require_replicated_projector_policy_config({"dtensor_cfg": {"expert_parallel_size": 8}})
    with pytest.raises(NotImplementedError, match="TP=1 and CP=1"):
        require_replicated_projector_policy_config({"dtensor_cfg": {"tensor_parallel_size": 2}})
    with pytest.raises(ValueError, match="cpu_offload"):
        require_replicated_projector_policy_config({"keep_policy_on_gpu": True, "dtensor_cfg": {"cpu_offload": True}})

    require_generation_topology(
        {
            "keep_vllm_on_gpu": True,
            "colocated": {"enabled": True},
            "vllm_cfg": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "expert_parallel_size": 1,
                "async_engine": False,
            },
        }
    )
    require_generation_topology({"vllm_cfg": {"tensor_parallel_size": 4}})
    require_generation_topology(
        {
            "vllm_cfg": {
                "tensor_parallel_size": 8,
                "expert_parallel_size": 8,
            }
        }
    )
    with pytest.raises(NotImplementedError, match="EP equals TP"):
        require_generation_topology({"vllm_cfg": {"expert_parallel_size": 2}})
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        require_generation_topology({"vllm_cfg": {"pipeline_parallel_size": 2}})


def test_explicit_vllm_executor_backend_overrides_nemo_default():
    """Reference inference may retain its qualified local executor topology."""
    kwargs = {"distributed_executor_backend": "ray"}
    _select_vllm_executor_backend(kwargs, "mp")
    assert kwargs["distributed_executor_backend"] == "mp"

    with pytest.raises(ValueError, match="must be 'mp' or 'ray'"):
        _select_vllm_executor_backend(kwargs, "uni")


def test_encoder_hf_overrides_preserve_native_mxfp8_config():
    quantization = {"quant_method": "modelopt", "quant_algo": "MXFP8"}
    llm_kwargs = {"hf_overrides": {"quantization_config": quantization}}

    _merge_encoder_hf_overrides(
        llm_kwargs,
        {"mm_image_num_tokens": 49},
        "LlavaMBridgeNemotronProjected",
    )

    assert llm_kwargs["hf_overrides"] == {
        "quantization_config": quantization,
        "mm_image_num_tokens": 49,
        "architectures": ["LlavaMBridgeNemotronProjected"],
    }


def test_encoder_hf_overrides_translate_the_training_architecture():
    # The config carries the training-side host class; the vLLM engine must
    # name the plugin-registered rollout class. Translated, not a conflict.
    llm_kwargs: dict = {}
    _merge_encoder_hf_overrides(
        llm_kwargs,
        {"architectures": ["LlavaExampleNemotronForCausalLM"], "mm_image_num_tokens": 49},
        "LlavaExampleNemotronProjected",
    )
    assert llm_kwargs["hf_overrides"]["architectures"] == ["LlavaExampleNemotronProjected"]
    assert llm_kwargs["hf_overrides"]["mm_image_num_tokens"] == 49


def test_encoder_hf_overrides_reject_conflicts():
    with pytest.raises(ValueError, match="conflicting vLLM HF override"):
        _merge_encoder_hf_overrides(
            {"hf_overrides": {"mm_image_num_tokens": 49}},
            {"mm_image_num_tokens": 50},
            "LlavaExampleNemotronProjected",
        )
