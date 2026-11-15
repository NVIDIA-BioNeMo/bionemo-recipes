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

"""Memory-bounded BF16 Evo2 inference."""

from functools import lru_cache
from pathlib import Path

import torch
from megatron.core.transformer.mlp import MLP


@lru_cache(maxsize=1)
def kernels():
    """Load the fused projection, convolution and rotary kernels."""
    from torch.utils.cpp_extension import load

    return load(
        name="evo2_inference",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "--fmad=true"],
    )


class HyenaMLP(MLP):
    """Tile inference activations while retaining native parameters and arithmetic."""

    def forward(self, hidden_states, per_token_scale=None, **kwargs):
        """Write each native MLP tile directly into its final output slice."""
        if (
            self.training
            or torch.is_grad_enabled()
            or self.config.fp8
            or self.config.tensor_model_parallel_size != 1
            or self.config.sequence_parallel
            or self.config.params_dtype != torch.bfloat16
            or hidden_states.shape[0] <= 16384
        ):
            return super().forward(hidden_states, per_token_scale=per_token_scale, **kwargs)

        output = None
        for start in range(0, hidden_states.shape[0], 16384):
            stop = start + 16384
            scale = None if per_token_scale is None else per_token_scale[start:stop]
            value, bias = super().forward(hidden_states[start:stop], per_token_scale=scale, **kwargs)
            if output is None:
                output = value.new_empty((hidden_states.shape[0], *value.shape[1:]))
            output[start:stop].copy_(value)
            del value
        return output, bias
