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

"""DTensor policy worker bridge for arbitrary encoder payloads.

Moved from ct-nemotron (ct-nemotron port Phase 4). Compact-PEFT warm start is
NeMo RL's own ``dtensor_cfg.lora_cfg.restore_from`` path (adopted from U-8 /
RL#3874); this module keeps the package's fuller provenance gate over the
donor adapter. The consumer's architecture registration is config-driven
(``policy.model_registry_callback``) so the package holds no modality names.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import torch

from nemotron_stitch.automodel.model import materialize_mm_projector
from nemotron_stitch.nemo_rl.transport import resolve_callback
from nemotron_stitch.projector.artifact import load_projector_artifact
from nemotron_stitch.provenance import require_provenance


def validate_initial_adapter_provenance(source: str | Path, expected_config: dict[str, Any]) -> None:
    """Check the donor adapter against the locked provenance before warm start.

    NeMo RL 4d969c93 loads ``dtensor_cfg.lora_cfg.restore_from`` inside policy
    setup through the model-family state-dict adapter, validating the donor's
    rank, scaling, and base model. The package's provenance lock can carry more
    than those three fields, and the upstream PEFT load is non-strict by the
    time this worker could inspect the result, so the check runs here — at
    construction, before any placement group is allocated.
    """
    if not expected_config:
        raise ValueError("initial_adapter_expected_config must not be empty")
    source = Path(source)
    adapter_dir = next(
        (candidate for candidate in (source, source / "model") if (candidate / "adapter_model.safetensors").is_file()),
        None,
    )
    if adapter_dir is None:
        raise FileNotFoundError(
            f"dtensor_cfg.lora_cfg.restore_from={str(source)!r}: no adapter_model.safetensors "
            "found there or in its 'model' subdirectory"
        )
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"SFT adapter config is missing: {config_path}")
    require_provenance(
        json.loads(config_path.read_text()),
        expected_config,
        context="SFT LoRA adapter",
    )


def require_replicated_projector_policy_config(config: dict[str, Any]) -> None:
    """Keep sidecar features local while upstream owns FSDP and expert sharding.

    Frozen projector replicas consume complete hidden vectors on each policy
    rank. Tensor/context parallel feature partitioning is not implemented;
    data and expert meshes leave this modality boundary local and unchanged.
    """
    dtensor = config.get("dtensor_cfg") or {}
    dimensions = {
        name: int(dtensor.get(name, 1))
        for name in (
            "tensor_parallel_size",
            "context_parallel_size",
            "expert_parallel_size",
            "dp_replicate_size",
        )
    }
    if any(value < 1 for value in dimensions.values()):
        raise ValueError(f"policy mesh dimensions must be positive: {dimensions}")
    if dimensions["tensor_parallel_size"] != 1 or dimensions["context_parallel_size"] != 1:
        raise NotImplementedError(f"encoder policy requires TP=1 and CP=1: {dimensions}")
    if config.get("keep_policy_on_gpu") and dtensor.get("cpu_offload"):
        raise ValueError("keep_policy_on_gpu is incompatible with dtensor_cfg.cpu_offload")


try:
    import ray
    from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import DTensorPolicyWorkerV2Impl

    @ray.remote(runtime_env=get_runtime_env_for_policy_worker("encoder_dtensor_policy_worker_v2"))
    class EncoderDTensorPolicyWorkerV2(DTensorPolicyWorkerV2Impl):
        """Add sidecar warm-start provenance and encoder payloads to NeMo RL's policy worker.

        NeMo RL 4d969c93 owns the compact-PEFT warm start and the worker
        extension seam; this subclass adds the package's donor-adapter
        provenance gate, the sidecar projector load, and external encoder
        payload handling.
        """

        def __init__(self, *args, **kwargs):
            positional = list(args)
            config = kwargs.get("config") or (positional[0] if positional else None)
            if config is None:
                raise TypeError("encoder policy worker requires a config mapping")
            registry_callback = config.get("model_registry_callback")
            if not registry_callback:
                raise ValueError("policy.model_registry_callback is required")
            resolve_callback(str(registry_callback))()
            require_replicated_projector_policy_config(config)
            # U-8: the warm start itself is upstream's — NeMo RL loads
            # dtensor_cfg.lora_cfg.restore_from inside super().__init__() and
            # ignores it when resuming a checkpoint. Only the provenance gate
            # is ours, and it must run before that load.
            initial_adapter = (config.get("dtensor_cfg") or {}).get("lora_cfg", {}).get("restore_from")
            supplied_weights = kwargs.get("weights_path")
            if len(positional) > 1:
                supplied_weights = positional[1]
            initial_sft_adapter = bool(initial_adapter) and supplied_weights is None
            if initial_sft_adapter:
                validate_initial_adapter_provenance(
                    initial_adapter,
                    config.get("initial_adapter_expected_config") or {},
                )
            self.keep_policy_on_gpu = bool(config.get("keep_policy_on_gpu", False))
            super().__init__(*positional, **kwargs)

            artifact = config.get("projector_artifact_path") if config is not None else None
            if not artifact:
                raise ValueError("policy.projector_artifact_path is required")
            expected_provenance = config.get("projector_expected_provenance")
            # The projector is project state outside the module tree (L-2); it
            # stays on meta after model construction. Materialize it before the
            # load and move it to the model device.
            materialize_mm_projector(self.model)
            load_projector_artifact(
                self.model,
                Path(artifact),
                expected=dict(expected_provenance) if expected_provenance else None,
            )
            projector_parameters = list(self.model.mm_projector.parameters())
            if not projector_parameters:
                raise RuntimeError("encoder policy has no projector parameters")
            if any(parameter.requires_grad for parameter in projector_parameters):
                raise RuntimeError("GRPO requires the projector to remain frozen")
            optimizer_ids = {id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]}
            if any(id(parameter) in optimizer_ids for parameter in projector_parameters):
                raise RuntimeError("projector parameters entered the GRPO optimizer")
            trainable = [
                (name, parameter.numel())
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            ]
            if self.rank == 0:
                print(
                    "ENCODER_POLICY_AUDIT "
                    + json.dumps(
                        {
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "trainable_parameters": sum(size for _, size in trainable),
                            "lora_tensors": sum("lora_" in name for name, _ in trainable),
                            "projector_trainable_tensors": sum(
                                parameter.requires_grad for parameter in projector_parameters
                            ),
                            "initial_sft_adapter": initial_sft_adapter,
                            "keep_policy_on_gpu": self.keep_policy_on_gpu,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        def _report_resident_memory(self, stage: str) -> None:
            if self.rank != 0:
                return
            gib = 1024**3
            print(
                "ENCODER_POLICY_MEMORY "
                + json.dumps(
                    {
                        "allocated_gib": round(torch.cuda.memory_allocated() / gib, 3),
                        "reserved_gib": round(torch.cuda.memory_reserved() / gib, 3),
                        "stage": stage,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        def prepare_for_lp_inference(self, *args, **kwargs) -> None:
            # U-9: NeMo RL e983576 always follows its offload lifecycle. This
            # opt-in GB300 path keeps the policy resident because policy + vLLM
            # fit together. Delete these overrides when upstream supports a
            # colocated residency policy. af0a11f added keep_train_buffers;
            # e983576's signature lacks it,
            # so accept and forward variadically like prepare_for_training.
            if not self.keep_policy_on_gpu:
                return super().prepare_for_lp_inference(*args, **kwargs)

            keep_train_buffers = bool(kwargs.get("keep_train_buffers", False))
            self.model = self.move_buffer_to_device(self.model, "cuda")
            self.model.eval()
            torch.randn(1, device="cuda")
            if self.optimizer is not None and self.offload_optimizer_for_logprob and not keep_train_buffers:
                self.move_optimizer_to_device("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            self._report_resident_memory("logprob")

        def prepare_for_training(self, *args, **kwargs) -> None:
            # Symmetric override for the same resident-policy lifecycle; all
            # other configurations delegate unchanged to NeMo RL.
            if not self.keep_policy_on_gpu:
                return super().prepare_for_training(*args, **kwargs)

            self.model = self.move_buffer_to_device(self.model, "cuda")
            self.model.train()
            if self.optimizer is not None and not self.cpu_offload:
                self.move_optimizer_to_device("cuda")
            torch.cuda.empty_cache()
            self._report_resident_memory("training")

        @torch.no_grad()
        def offload_after_refit(self) -> None:
            if not self.keep_policy_on_gpu:
                return super().offload_after_refit()

            # A GB300 can hold the BF16 policy beside the colocated vLLM engine.
            # Keep the expensive base parameters resident and move only optimizer
            # state while vLLM owns the compute phase.
            self.model = self.move_buffer_to_device(self.model, "cuda")
            self.model.eval()
            self.offload_before_refit()
            self._report_resident_memory("after_refit")

except ImportError:
    EncoderDTensorPolicyWorkerV2 = None
