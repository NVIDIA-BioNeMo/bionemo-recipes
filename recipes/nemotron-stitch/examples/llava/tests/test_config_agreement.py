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

"""Keep the self-contained model stage configs aligned with application constants."""

from pathlib import Path
from types import SimpleNamespace

import llava_qwen
import pytest
import yaml

CONFIGS = Path(__file__).parents[1] / "configs"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text())


def test_qwen_automodel_configs_agree():
    for name in ("alignment-qwen.yaml", "sft-qwen.yaml"):
        config = _load(name)
        model = config["model"]
        projector = model["mm_projectors"][0]

        assert model["architectures"] == [llava_qwen.TRAINING_ARCHITECTURE]
        assert model["backend"] == {
            "_target_": "nemo_automodel.components.models.common.BackendConfig",
            "attn": "sdpa",
        }
        assert model["attn_implementation"] == "flash_attention_2"
        assert projector["name"] == llava_qwen.PROJECTOR_NAME
        assert model["mm_placeholder_token_ids"] == {llava_qwen.PROJECTOR_NAME: 248055}
        assert config["step_scheduler"]["local_batch_size"] == 1
        assert config["packed_sequence"] == {
            "packed_sequence_size": 2048,
            "packing_strategy": "nemotron_stitch.automodel.data.StreamingFeaturePackingConfig",
            "packing_format": "neat",
        }
        assert config["dataset"]["_target_"] == ("nemotron_stitch.automodel.data.ManifestFeatureIterableDatasetConfig")
        assert config["dataset"]["projector_name"] == llava_qwen.PROJECTOR_NAME
        assert config["dataset"]["placeholder_token_id"] == 248055
        assert config["dataloader"]["batch_size"] is None
        assert config["dataloader"]["shuffle_buffer_size"] == 1000


def test_lightning_alignment_and_sft_stream_feature_preserving_thd_packs():
    for name in ("alignment-lightning.yaml", "sft-lightning.yaml"):
        config = _load(name)
        packing = config["packed_sequence"]
        dataset = config["dataset"]

        assert config["model"]["backend"] == {
            "_target_": "nemo_automodel.components.models.common.BackendConfig",
            "attn": "te",
        }
        assert config["step_scheduler"]["local_batch_size"] == 1
        assert packing == {
            "packed_sequence_size": 2048,
            "packing_strategy": "nemotron_stitch.automodel.data.StreamingFeaturePackingConfig",
            "packing_format": "thd",
        }
        assert dataset["_target_"] == "nemotron_stitch.automodel.data.ManifestFeatureIterableDatasetConfig"
        assert config["dataloader"]["batch_size"] is None
        assert config["dataloader"]["shuffle_buffer_size"] == 1000


def test_qwen_grpo_configs_agree():
    model = _load("models/qwen3.5-4b.yaml")["policy"]
    run = _load("grpo-qwen.yaml")

    assert model["hf_config_overrides"]["architectures"] == [llava_qwen.TRAINING_ARCHITECTURE]
    assert model["hf_config_overrides"]["mm_encoder_start"] == "<|vision_start|>"
    assert model["hf_config_overrides"]["mm_encoder_placeholder"] == "<|vision_pad|>"
    assert model["hf_config_overrides"]["mm_encoder_end"] == "<|vision_end|>"
    assert run["policy"]["generation"]["mm_plugin_callback"] == "llava_qwen.register_vllm"
    assert run["policy"]["generation"]["mm_architecture"] == llava_qwen.ROLLOUT_ARCHITECTURE


@pytest.mark.parametrize(
    "name",
    (
        "grpo.yaml",
        "grpo-qwen.yaml",
        "grpo-lightning.yaml",
        "grpo-qwen3.6-35b.yaml",
    ),
)
def test_colocated_grpo_keeps_both_workers_resident(name):
    run = _load(name)

    assert run["loss_fn"]["force_on_policy_ratio"] is True
    rollout_batch = run["grpo"]["num_prompts_per_step"] * run["grpo"]["num_generations_per_prompt"]
    assert run["policy"]["train_global_batch_size"] == rollout_batch
    assert run["policy"]["generation_batch_size"] == rollout_batch
    assert run["policy"]["keep_policy_on_gpu"] is True
    generation = run["policy"]["generation"]
    assert generation["keep_vllm_on_gpu"] is True
    assert generation["colocated"]["enabled"] is True
    assert generation["vllm_kwargs"]["kv_cache_memory_bytes"] == 4 * 1024**3
    assert generation["vllm_cfg"]["enforce_eager"] is False
    assert generation["vllm_kwargs"]["compilation_config"] == {
        "mode": "none",
        "cudagraph_mode": "full_decode_only",
    }
    assert generation["vllm_kwargs"]["max_num_seqs"] == "${policy.generation_batch_size}"


def test_two_gpu_grpo_uses_upstream_non_colocated_lifecycle():
    run = _load("grpo-2gpu.yaml")

    assert run["policy"]["keep_policy_on_gpu"] is False
    assert run["policy"]["generation"]["keep_vllm_on_gpu"] is False
    assert run["policy"]["generation"]["colocated"]["enabled"] is False


def test_qwen36_moe_configs_agree():
    for name in ("alignment-qwen3.6-35b.yaml", "sft-qwen3.6-35b.yaml"):
        config = _load(name)
        model = config["model"]

        assert model["architectures"] == [llava_qwen.MOE_TRAINING_ARCHITECTURE]
        assert model["backend"]["experts"] == "torch_mm"
        assert model["backend"]["dispatcher"] == "torch"
        assert model["mm_placeholder_token_ids"] == {llava_qwen.PROJECTOR_NAME: 248055}
        assert config["step_scheduler"]["max_steps"] == 1
        assert config["packed_sequence"]["packing_strategy"] == (
            "nemotron_stitch.automodel.data.StreamingFeaturePackingConfig"
        )
        assert config["packed_sequence"]["packing_format"] == "neat"
        assert config["dataset"]["_target_"] == ("nemotron_stitch.automodel.data.ManifestFeatureIterableDatasetConfig")
        assert config["dataloader"]["batch_size"] is None
        assert config["dataloader"]["shuffle_buffer_size"] == 1000
        assert config["projector"]["model_registry_callback"] == "llava_qwen.register_moe_models"

    assert _load("sft-qwen3.6-35b.yaml")["peft"]["target_modules"] == [
        "model.language_model.layers.*.self_attn.q_proj",
        "model.language_model.layers.*.self_attn.k_proj",
        "model.language_model.layers.*.self_attn.v_proj",
        "model.language_model.layers.*.self_attn.o_proj",
    ]

    model = _load("models/qwen3.6-35b-a3b.yaml")["policy"]
    run = _load("grpo-qwen3.6-35b.yaml")
    assert model["hf_config_overrides"]["architectures"] == [llava_qwen.MOE_TRAINING_ARCHITECTURE]
    assert run["policy"]["generation"]["mm_plugin_callback"] == "llava_qwen.register_moe_vllm"
    assert run["policy"]["generation"]["mm_architecture"] == llava_qwen.MOE_ROLLOUT_ARCHITECTURE
    assert run["grpo"]["max_num_steps"] == 1


def test_qwen_vllm_accepts_external_embeddings_without_native_media():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm import ModelRegistry

    llava_qwen.register_vllm()
    model_cls = ModelRegistry._try_load_model_cls(llava_qwen.ROLLOUT_ARCHITECTURE)
    model = object.__new__(model_cls)
    payload = torch.zeros(49, 2560)

    embeddings = model.embed_multimodal(clip_image_embeds=payload)

    assert len(embeddings) == 1
    assert embeddings[0] is payload


def test_qwen_moe_vllm_accepts_external_embeddings_without_native_media():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm import ModelRegistry

    llava_qwen.register_moe_vllm()
    model_cls = ModelRegistry._try_load_model_cls(llava_qwen.MOE_ROLLOUT_ARCHITECTURE)
    model = object.__new__(model_cls)
    payload = torch.zeros(49, 2048)

    embeddings = model.embed_multimodal(clip_image_embeds=payload)

    assert len(embeddings) == 1
    assert embeddings[0] is payload


def test_qwen_vllm_excludes_external_embeddings_from_native_mrope(monkeypatch):
    pytest.importorskip("vllm")
    from vllm import ModelRegistry
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    observed = []

    def get_positions(self, input_tokens, mm_features):
        del self, input_tokens
        observed.extend(feature.modality for feature in mm_features)
        return "positions"

    monkeypatch.setattr(Qwen3_5ForConditionalGeneration, "get_mrope_input_positions", get_positions)
    llava_qwen.register_vllm()
    model_cls = ModelRegistry._try_load_model_cls(llava_qwen.ROLLOUT_ARCHITECTURE)
    model = object.__new__(model_cls)

    result = model.get_mrope_input_positions(
        [1, 2, 3],
        [SimpleNamespace(modality="clip_image"), SimpleNamespace(modality="image")],
    )

    assert result == "positions"
    assert observed == ["image"]
