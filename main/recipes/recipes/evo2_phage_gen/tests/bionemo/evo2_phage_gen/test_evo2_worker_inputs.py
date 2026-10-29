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


"""Regressions for replay inputs in independently imported policy workers."""

import subprocess
import sys
import textwrap

import pytest
import torch


def test_fresh_worker_maskless_inputs():
    """The native adapter loader must install the fix without running the driver."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import sys
            from unittest.mock import patch
            import torch
            from nemo_rl.distributed.batched_data_dict import BatchedDataDict
            from nemo_rl.models.megatron import data
            from nemo_rl.models.generation.megatron.megatron_generation import _load_generation_adapter

            assert 'bionemo.evo2_phage_gen.run_phage_grpo' not in sys.modules
            _load_generation_adapter({'mcore_generation_config': {
                'generation_adapter': 'bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter',
                'generation_adapter_config': {},
            }})
            for length in (7, 91136):
                ids = torch.full((1, length), 65, dtype=torch.long)
                ids[0, -2:] = 0  # Authentic EOD and right padding keep their absolute positions.
                actions = torch.zeros_like(ids, dtype=torch.float32)
                actions[0, length // 2:-1] = 1
                batch = BatchedDataDict(input_ids=ids, token_mask=actions)
                # CP1 topology only; keep the real native unpacked preprocessing path.
                with patch.object(data, 'get_context_parallel_rank', return_value=0), \\
                     patch.object(data, 'get_context_parallel_world_size', return_value=1):
                    processed = data.process_microbatch(batch, pack_sequences=False)
                assert processed.attention_mask is None, 'Worker built a dense attention mask'
                torch.testing.assert_close(processed.position_ids, torch.arange(length).unsqueeze(0))
                assert processed.input_ids is ids
                assert batch['token_mask'] is actions
            print('FRESH_WORKER_MASKLESS_INPUTS_PASS')
        """),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FRESH_WORKER_MASKLESS_INPUTS_PASS" in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU for Transformer Engine causal attention")
def test_implicit_causal_parity():
    """Omitting the redundant mask preserves causal outputs and gradients."""
    from transformer_engine.pytorch import DotProductAttention

    torch.manual_seed(17)
    attention = DotProductAttention(
        num_attention_heads=4,
        kv_channels=16,
        attention_dropout=0.0,
        qkv_format="sbhd",
        attn_mask_type="causal",
    ).cuda()
    qkv = [torch.randn(257, 1, 4, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    mask = torch.ones(1, 1, 257, 257, device="cuda", dtype=torch.bool).triu(1)
    explicit = attention(*qkv, attention_mask=mask)
    expected_grad = torch.autograd.grad(explicit.float().square().mean(), qkv)
    implicit = attention(*qkv, attention_mask=None)
    actual_grad = torch.autograd.grad(implicit.float().square().mean(), qkv)
    torch.testing.assert_close(implicit, explicit, atol=2e-2, rtol=2e-2)
    for actual, expected in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-2)
    changed = [value.detach().clone() for value in qkv]
    changed[1][128:] *= -3
    changed[2][128:] += 10
    perturbed = attention(*changed, attention_mask=None)
    torch.testing.assert_close(perturbed[:128], implicit[:128], atol=2e-2, rtol=2e-2)
