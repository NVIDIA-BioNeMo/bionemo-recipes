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

"""Bound Transformer Engine normalization calls during Evo2 inference."""

import torch
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TENorm,
    _get_extra_te_kwargs,
)
from transformer_engine.pytorch import RMSNorm


def _inference_tiling(module, hidden_states):
    """Bound normalization only for unsharded BF16 inference."""
    config = module.config
    return (
        not module.training
        and not torch.is_grad_enabled()
        and not config.fp8
        and config.tensor_model_parallel_size == 1
        and not config.sequence_parallel
        and config.params_dtype == torch.bfloat16
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.shape[0] > 16384
    )


def _tile_rows(forward, hidden_states):
    """Call the original operator on fixed 16K rows and assemble its output."""
    output = None
    tail = None
    for start in range(0, hidden_states.shape[0], 16384):
        tile = hidden_states[start : start + 16384]
        valid = tile.shape[0]
        if valid < 16384:
            tile = torch.nn.functional.pad(tile, (0, 0) * (tile.ndim - 1) + (0, 16384 - valid))
        result = forward(tile)
        value = result[0] if isinstance(result, tuple) else result
        if output is None:
            output = value.new_empty((hidden_states.shape[0], *value.shape[1:]))
        output[start : start + valid].copy_(value[:valid])
        tail = result[1:] if isinstance(result, tuple) else None
    return (output, *tail) if tail is not None else output


class HyenaNormLinear(TELayerNormColumnParallelLinear):
    """Keep native checkpoint keys and bound normalization/linear workspaces."""

    def forward(self, x):
        """Preserve native training and explicit normalization-output behavior."""
        if self.return_layernorm_output or not _inference_tiling(self, x):
            return super().forward(x)
        return _tile_rows(super().forward, x)


class HyenaRMSNorm(RMSNorm):
    """Use the same bounded normalization call size as the fused linears."""

    def __init__(self, config, hidden_size, eps):
        """Preserve native initialization and checkpoint parameter names."""
        super().__init__(
            normalized_shape=hidden_size,
            eps=eps,
            sequence_parallel=config.sequence_parallel,
            zero_centered_gamma=config.layernorm_zero_centered_gamma,
            **_get_extra_te_kwargs(config),
        )
        self.config = config

    def forward(self, inp):
        """Keep the full native call outside BF16 inference."""
        if not _inference_tiling(self, inp):
            return super().forward(inp)
        return _tile_rows(super().forward, inp)


def get_hyena_norm(config, hidden_size, eps):
    """Retain the native LayerNorm fallback and RMSNorm parameter layout."""
    if config.normalization != "RMSNorm":
        return TENorm(config=config, hidden_size=hidden_size, eps=eps)
    return HyenaRMSNorm(config, hidden_size, eps)
