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

from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.mlp import MLP

from bionemo.evo2.models.megatron.hyena.inference import HyenaMLP, kernels
from bionemo.evo2.models.megatron.hyena.packed_kernels import (
    local_positions_from_cu_seqlens,
    modal_chunk_metadata_from_cu_seqlens,
    modal_poles,
    segmented_causal_conv1d,
    segmented_fir_from_projection,
    segmented_modal_from_projection,
    sequence_ids_from_cu_seqlens,
)
from bionemo.evo2.models.megatron.hyena.packed_rope import (
    _launch_packed_rope,
    apply_preindexed_fused_rope_thd,
)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _assert_same_bits(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.parametrize("length", [1, 16384, 16385, 32771])
@pytest.mark.parametrize("scaled", [False, True])
def test_mlp_tiles(monkeypatch, length, scaled):
    torch.manual_seed(13)
    model = HyenaMLP.__new__(HyenaMLP)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        fp8=None, tensor_model_parallel_size=1, sequence_parallel=False, params_dtype=torch.bfloat16
    )
    model.first = torch.nn.Linear(7, 22, bias=False)
    model.second = torch.nn.Linear(11, 7, bias=False)
    calls = []

    def native(self, value, per_token_scale=None, **kwargs):
        calls.append(value.shape[0])
        gate, drive = self.first(value).chunk(2, dim=-1)
        result = self.second(torch.nn.functional.gelu(gate) * drive)
        if per_token_scale is not None:
            result = result * per_token_scale[..., None]
        return result, None

    monkeypatch.setattr(MLP, "forward", native)
    inputs = torch.randn(length, 2, 7)
    scale = torch.rand(length, 2) if scaled else None
    state = {name: value.clone() for name, value in model.state_dict().items()}
    model.eval()
    with torch.inference_mode():
        expected, _ = native(model, inputs, scale)
        calls.clear()
        actual, bias = model(inputs, scale)
    torch.testing.assert_close(actual, expected)
    assert bias is None and max(calls) <= 16384
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])

    calls.clear()
    model.train()
    output, _ = model(inputs.requires_grad_(), scale)
    output.sum().backward()
    assert calls == [length] and torch.isfinite(inputs.grad).all()


def _inputs(channels, group_width):
    lengths = [1, 17, 65, 257]
    boundaries = torch.tensor([0, 1, 18, 83, 340], device="cuda", dtype=torch.int32)
    projection = torch.randn(sum(lengths), 3 * channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(3 * channels // group_width, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    positions = local_positions_from_cu_seqlens(boundaries, sum(lengths))
    sequence_ids = sequence_ids_from_cu_seqlens(boundaries, sum(lengths))
    return projection, weight, boundaries, positions, sequence_ids


@cuda
@pytest.mark.parametrize("channels,group_width", [(33, 1), (66, 3), (128, 4)])
@pytest.mark.parametrize("taps", [7, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_diagonal", [False, True])
@torch.inference_mode()
def test_fused_fir(channels, group_width, taps, dtype, with_diagonal):
    torch.manual_seed(17)
    projection, weight, _, positions, sequence_ids = _inputs(channels, group_width)
    mixer = torch.randn(channels // group_width, taps, device="cuda", dtype=dtype) * 0.1
    diagonal = torch.randn(channels, device="cuda", dtype=torch.bfloat16) if with_diagonal else None
    filtered = segmented_causal_conv1d(projection, weight, sequence_ids, group_width=group_width)
    expected = segmented_fir_from_projection(
        filtered, mixer, positions, group_width=group_width, flip_filter=taps == 7, diagonal=diagonal
    )
    actual = kernels().fused_fir(
        projection, weight, mixer, diagonal, sequence_ids, positions, group_width, group_width
    )
    _assert_same_bits(actual, expected)


@cuda
@pytest.mark.parametrize("chunk_size", [None, 32, 128])
@pytest.mark.parametrize("group_width", [1, 3])
@torch.inference_mode()
def test_compact_modal(chunk_size, group_width):
    torch.manual_seed(19)
    channels = 66
    projection, weight, boundaries, _, sequence_ids = _inputs(channels, group_width)
    filtered = segmented_causal_conv1d(projection, weight, sequence_ids, group_width=group_width)
    gate, drive = kernels().compact_projection(projection, weight, sequence_ids, group_width)
    _assert_same_bits(gate, filtered[:, 0::3])
    _assert_same_bits(drive, filtered[:, 1::3] * filtered[:, 2::3])
    diagonal = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    residues = torch.randn(channels // group_width, 16, device="cuda", dtype=torch.bfloat16)
    gamma = torch.full_like(residues, -5)
    poles = modal_poles(gamma, torch.randn_like(residues))
    chunks = None if chunk_size is None else modal_chunk_metadata_from_cu_seqlens(boundaries, chunk_size=chunk_size)
    before_drive = drive.clone()
    expected = segmented_modal_from_projection(
        filtered, diagonal, residues, poles, boundaries, group_width=group_width, chunk_metadata=chunks
    )
    actual = segmented_modal_from_projection(
        (gate, drive), diagonal, residues, poles, boundaries, group_width=group_width, chunk_metadata=chunks
    )
    assert actual.data_ptr() == gate.data_ptr()
    assert torch.equal(drive, before_drive)
    _assert_same_bits(actual, expected)


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("rotary_dim", [32, 64])
@torch.inference_mode()
def test_inplace_rope(dtype, interleaved, rotary_dim):
    torch.manual_seed(23)
    qkv = torch.randn(37, 3, 5, 64, device="cuda", dtype=torch.bfloat16)
    before = qkv.clone()
    tokens = qkv[:, 0]
    angle = torch.randn(37, 1, 1, rotary_dim, device="cuda")
    frequencies = torch.cat((angle.cos(), angle.sin()), dim=-2).to(dtype)
    expected = _launch_packed_rope(tokens, frequencies, interleaved=interleaved, inverse=False)
    boundaries = torch.tensor([0, 17, 37], device="cuda", dtype=torch.int32)
    actual = apply_preindexed_fused_rope_thd(tokens, boundaries, frequencies, interleaved=interleaved)
    assert actual.data_ptr() == tokens.data_ptr()
    assert torch.equal(qkv[:, 1:], before[:, 1:])
    _assert_same_bits(actual, expected)


@cuda
@pytest.mark.parametrize("length", [16384, 16385, 32771])
@pytest.mark.parametrize("width", [1920, 4096, 5120])
@torch.inference_mode()
def test_norm_tiles(length, width):
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformer_engine.pytorch import RMSNorm

    from bionemo.evo2.models.megatron.hyena.inference_norm import HyenaRMSNorm

    config = TransformerConfig(
        num_layers=1,
        hidden_size=width,
        num_attention_heads=1,
        normalization="RMSNorm",
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    torch.manual_seed(29)
    norm = HyenaRMSNorm(config, width, 1e-6).cuda().eval()
    values = torch.randn(length, 1, width, device="cuda", dtype=torch.bfloat16)
    expected = RMSNorm.forward(norm, values)
    actual = norm(values)
    if length <= 16384:
        _assert_same_bits(actual, expected)
    for start in range(0, length, 1024):
        value = values[start : start + 1024].double()
        oracle = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * norm.weight.double()
        oracle = oracle.to(torch.bfloat16)
        for result in (expected, actual):
            torch.testing.assert_close(
                result[start : start + 1024], oracle, rtol=torch.finfo(torch.bfloat16).eps, atol=0
            )


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("logging", [False, True])
def test_embedding_release(monkeypatch, training, grad_enabled, logging):
    import weakref

    from megatron.core.utils import WrappedTensor

    from bionemo.evo2.models.megatron.hyena.hyena_model import HyenaModel

    model = HyenaModel.__new__(HyenaModel)
    torch.nn.Module.__init__(model)
    model.train(training)
    model.pre_process = True
    model.position_embedding_type = "none"
    model.config = SimpleNamespace(enable_cuda_graph=False, flash_decode=False)
    references = []

    def embedding(**kwargs):
        value = torch.randn(4, 1, 8)
        references.append(weakref.ref(value))
        return value

    model.embedding = embedding
    monkeypatch.setattr(
        "bionemo.evo2.models.megatron.hyena.hyena_model.has_config_logger_enabled", lambda config: logging
    )
    with torch.set_grad_enabled(grad_enabled):
        result = model._preprocess(torch.zeros(1, 4, dtype=torch.long), torch.arange(4)[None])[0]
    if training or grad_enabled or logging:
        assert result is references[0]()
    else:
        assert isinstance(result, WrappedTensor)
        value = result.unwrap()
        assert value is references[0]()
        del value
        assert references[0]() is None
