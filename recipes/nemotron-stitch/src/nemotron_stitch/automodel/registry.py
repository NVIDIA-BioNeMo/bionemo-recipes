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

"""Idempotent out-of-tree architecture registration and aliases.

Moved from ct-nemotron's ``model/registry.py`` (ct-nemotron port Phase 3),
generalized from the conditioned-Omni architecture to any ``(name, class)``
pair. Holds the private ``ModelRegistry`` import that upstream U-2 deletes —
see docs/upstream-gaps.md.
"""

from __future__ import annotations

from typing import Any


def register_models(architecture: str, model_cls: type, registry: Any | None = None) -> bool:
    """Register ``model_cls`` under ``architecture`` in AutoModel's model registry, once.

    Re-registering the same class under the same name is a no-op; a
    conflicting registration fails closed. Returns ``False`` when AutoModel is
    not installed, so data-only environments can still import consumer modules
    (lazy framework import, design §3.7).
    """
    if registry is None:
        try:
            # NeMo AutoModel 24b47e856263d313b942f0ed666c63fff83306b4 resolves
            # architectures through this private registry and has no public
            # out-of-tree architecture registration hook. Delete when upstream
            # U-2 is adopted.
            from nemo_automodel._transformers.registry import ModelRegistry
        except ModuleNotFoundError:
            return False
        registry = ModelRegistry
    mapping = getattr(registry, "model_arch_name_to_cls", None)
    if mapping is not None and architecture in mapping:
        existing = mapping[architecture]
        if existing is model_cls:
            return True
        raise RuntimeError(f"AutoModel architecture {architecture!r} is already registered to {existing!r}")
    registry.register(architecture, model_cls, exist_ok=False)
    return True
