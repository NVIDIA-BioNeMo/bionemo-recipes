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

# --- BEGIN COPIED FILE NOTICE ---
# This file is copied from: recipes/evo2_megatron/src/bionemo/common/megatron_compat.py
# Do not modify this file directly. Instead, modify the source and run:
#     python ci/scripts/check_copied_files.py --fix
# --- END COPIED FILE NOTICE ---

"""Compatibility shims for the Megatron-Core release pinned by the Megatron recipes."""

import inspect
from pathlib import Path

import torch


def patch_megatron_load_tensors_metadata() -> None:
    """Make Megatron-Core's dist-checkpoint metadata loader tolerate new ``TensorProperties`` fields.

    PyTorch in the 26.10 NGC container (pytorch/pytorch#194251) added ``TensorProperties.strides``.
    The Megatron-Core bundled with Megatron-Bridge < v0.7.0 forwards ``**tp.properties.__dict__`` to
    ``torch.empty``, which rejects ``strides``, so every checkpoint load fails. This applies the upstream
    fix (NVIDIA/Megatron-LM#7620: pass only ``dtype``). Remove once the recipes pin Megatron-Bridge >= v0.7.0.
    """
    try:
        from megatron.core.dist_checkpointing.strategies import torch as mcore_torch_strategy
    except ImportError:
        return

    strategy_cls = mcore_torch_strategy.TorchDistLoadShardedStrategy
    try:
        needs_patch = "tp.properties.__dict__" in inspect.getsource(strategy_cls.load_tensors_metadata)
    except (OSError, TypeError):
        needs_patch = False
    if not needs_patch:
        return

    def load_tensors_metadata(self, checkpoint_dir: Path, metadata=None):
        """Uses tensors metadata stored in the metadata file."""
        if metadata is None:
            metadata = mcore_torch_strategy._get_filesystem_reader(checkpoint_dir).read_metadata()

        sharded_metadata = {}
        for k, tp in metadata.state_dict_metadata.items():
            if not isinstance(tp, mcore_torch_strategy.TensorStorageMetadata):
                continue  # load only tensors
            sharded_metadata[k] = mcore_torch_strategy.ShardedTensor.from_rank_offsets(
                k, torch.empty(tp.size, dtype=tp.properties.dtype, device="meta")
            ).without_data()
        return sharded_metadata

    strategy_cls.load_tensors_metadata = load_tensors_metadata
