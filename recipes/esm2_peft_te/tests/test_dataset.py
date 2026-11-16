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

import pytest

from dataset import create_dataloader
from distributed_config import DistributedConfig


@pytest.mark.parametrize("use_sequence_packing", [False, True])
def test_dataloader_workers_use_spawn(recipe_path, use_sequence_packing):
    """Forked workers can inherit CUDA state (e.g. a DDP Reducer) from the parent and abort when it is destroyed."""
    train_dataloader, _, _ = create_dataloader(
        distributed_config=DistributedConfig(),
        use_sequence_packing=use_sequence_packing,
        tokenizer_name=str(recipe_path / "example_8m_checkpoint"),
        micro_batch_size=2,
        val_micro_batch_size=2,
        num_workers=1,
        max_seq_length=128,
        stride=16,
        seed=42,
        ss3_classification=True,
        load_dataset_kwargs={
            "path": "parquet",
            "split": "train",
            "data_files": str(recipe_path / "data" / "peft_sanity_dataset.parquet"),
            "streaming": True,
        },
    )

    assert train_dataloader.multiprocessing_context.get_start_method() == "spawn"

    batch = next(iter(train_dataloader))
    assert batch["input_ids"].shape == batch["labels"].shape


@pytest.mark.parametrize("use_sequence_packing", [False, True])
@pytest.mark.parametrize("val_micro_batch_size", [1, 3])
def test_validation_batch_size(recipe_path, tmp_path, use_sequence_packing, val_micro_batch_size):
    """Validation must use its own batch size in both padded and packed formats."""
    from datasets import Dataset

    # Six residues plus the tokenizer's CLS and EOS fill an eight-token sample.
    data_file = tmp_path / "sequences.parquet"
    Dataset.from_dict({"Sequence": ["AAAAAA"] * 12, "Secondary_structure": ["HHHHHH"] * 12}).to_parquet(data_file)
    train_dataloader, val_dataloader, _ = create_dataloader(
        distributed_config=DistributedConfig(),
        use_sequence_packing=use_sequence_packing,
        tokenizer_name=str(recipe_path / "example_8m_checkpoint"),
        micro_batch_size=2,
        val_micro_batch_size=val_micro_batch_size,
        num_workers=0,
        max_seq_length=8,
        stride=0,
        seed=42,
        ss3_classification=True,
        load_dataset_kwargs={
            "path": "parquet",
            "data_files": {"train": str(data_file), "validation": str(data_file)},
            "streaming": True,
        },
    )

    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))
    assert train_batch["input_ids"].numel() == 2 * 8
    assert val_batch["input_ids"].numel() == val_micro_batch_size * 8
    assert val_batch["input_ids"].shape == val_batch["labels"].shape
