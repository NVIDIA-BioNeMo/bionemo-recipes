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

import torch
from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy
from torch.distributed.checkpoint.metadata import Metadata, TensorProperties, TensorStorageMetadata

import bionemo.evo2  # noqa: F401  (applies the Megatron-Core compatibility patch)


def test_load_tensors_metadata_ignores_tensor_strides(tmp_path):
    """``strides`` (pytorch/pytorch#194251) is a TensorProperties field that torch.empty rejects."""
    properties = TensorProperties(dtype=torch.bfloat16)
    properties.strides = (4, 1)
    metadata = Metadata(
        state_dict_metadata={"keyA": TensorStorageMetadata(properties=properties, size=torch.Size([3, 4]), chunks=[])}
    )

    sharded_metadata = TorchDistLoadShardedStrategy().load_tensors_metadata(tmp_path, metadata)

    assert sharded_metadata["keyA"].dtype == torch.bfloat16
    assert sharded_metadata["keyA"].global_shape == (3, 4)
