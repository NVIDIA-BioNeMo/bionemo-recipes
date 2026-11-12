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

"""Run the model code published to the `nvidia/esm2_*` Hugging Face Hub repos against the installed libraries.

Each Hub repo bundles its own copy of the model code (`esm_nv.py`), which can break when transformers or Transformer
Engine change even though the repo's `modeling_esm_te.py` is fine. These tests download only the config, remote
code, and tokenizer (no weights), build a small randomly initialized model from the Hub config, and run a padded
forward and backward pass.

They run on the nightly scheduled CI, or locally with `ESM2_TEST_HUB_CHECKPOINTS=1`.
"""

import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

from export import ESM_TAGS


pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_EVENT_NAME") != "schedule" and os.environ.get("ESM2_TEST_HUB_CHECKPOINTS") != "1",
    reason="Hub checkpoint tests run on the nightly schedule or with ESM2_TEST_HUB_CHECKPOINTS=1.",
)


@pytest.mark.parametrize("tag", ESM_TAGS)
def test_hub_remote_code_forward_backward(tag, test_proteins):
    repo_id = f"nvidia/{tag}"
    config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
    assert type(config).__name__ == "NVEsmConfig"

    # Shrink the model so no weights need to be downloaded or materialized; the remote code path is unchanged.
    config.num_hidden_layers = 2
    config.hidden_size = 320
    config.intermediate_size = 1280
    config.num_attention_heads = 20

    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True, dtype=torch.bfloat16).cuda()
    assert type(model).__name__ == "NVEsmForMaskedLM"

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15, pad_to_multiple_of=32, seed=42
    )
    batch = collator([tokenizer(p) for p in test_proteins])
    batch = {k: v.cuda() for k, v in batch.items()}
    assert not batch["attention_mask"].all(), "Inputs should include padding to exercise the attention mask."

    outputs = model(**batch)
    assert outputs.logits.shape[:2] == batch["input_ids"].shape
    assert torch.isfinite(outputs.loss)

    outputs.loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
