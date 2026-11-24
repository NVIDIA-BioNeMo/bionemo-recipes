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

"""Typed BioNeMo Perception Container clients and model-neutral tensor-response parsing."""

from .client import (
    ESM3Client,
    ESM3ForwardRequest,
    Evo2Client,
    Evo2ForwardRequest,
    ForwardRequest,
    KERMTClient,
    KermtForwardRequest,
    OutputSpec,
    PerceptionClient,
    PerceptionRequestError,
    PoolingSpec,
)
from .protocol import (
    CONTENT_TYPE,
    BFloat16Tensor,
    ForwardResponse,
    ProtocolError,
    WireTensor,
    parse_forward_response,
    parse_response,
)

__all__ = [
    "PerceptionClient",
    "ESM3ForwardRequest",
    "ESM3Client",
    "Evo2ForwardRequest",
    "Evo2Client",
    "ForwardRequest",
    "KermtForwardRequest",
    "KERMTClient",
    "PerceptionRequestError",
    "OutputSpec",
    "PoolingSpec",
    "CONTENT_TYPE",
    "BFloat16Tensor",
    "WireTensor",
    "ForwardResponse",
    "ProtocolError",
    "parse_forward_response",
    "parse_response",
]
