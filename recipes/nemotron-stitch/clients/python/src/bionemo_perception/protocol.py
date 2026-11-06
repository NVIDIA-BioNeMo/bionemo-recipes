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

"""Checked, zero-copy parser for BioNeMo Perception Container tensor envelopes.

Every model container emits a single version-1 binary envelope (BNMOFWD1). The parser
validates the envelope framing, its XXH64 checksum, and exposes zero-copy
tensor views. The envelope dtype is ``bfloat16-le`` (default transport) or
``float32-le`` (KERMT's full-precision mode, ``KERMT_WIRE_DTYPE=fp32``).
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import xxhash

# --- wire constants ---
CONTENT_TYPE = "application/vnd.bnmo.tensor+binary; version=1"
FORWARD_MAGIC = b"BNMOFWD1"
FORMAT_VERSION = 1
LEGACY_V2_MAGIC = b"BNMOFWD2"
DTYPE = "bfloat16-le"
DTYPE_F32 = "float32-le"
#: element size in bytes per wire dtype
DTYPE_ELT_SIZE = {DTYPE: 2, DTYPE_F32: 4}
PREAMBLE_BYTES = 24
PAYLOAD_ALIGNMENT = 64


class ProtocolError(ValueError):
    """A response does not satisfy the BioNeMo Perception Container tensor contract."""


@dataclass(frozen=True)
class WireTensor:
    """A zero-copy view of one row-major, little-endian tensor."""

    name: str
    shape: tuple[int, ...]
    data: memoryview
    dtype: str = DTYPE

    def numpy_float32(self) -> Any:
        """Return FP32 values without requiring PyTorch.

        BF16 is exactly widened into a new array; float32-le is a zero-copy
        view whose lifetime is tied to the original response buffer.
        """
        try:
            import numpy as np
        except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install bionemo-perception-client[numpy] for NumPy tensor views") from error
        if self.dtype == DTYPE_F32:
            return np.frombuffer(self.data, dtype="<f4").reshape(self.shape)
        if self.dtype != DTYPE:
            raise ProtocolError(f"unsupported tensor dtype: {self.dtype}")
        words = np.frombuffer(self.data, dtype="<u2").astype("<u4")
        words <<= 16
        return words.view("<f4").reshape(self.shape)

    def torch_bfloat16(self) -> Any:
        """Return a zero-copy ``torch.bfloat16`` tensor view (BF16 envelopes only).

        The returned tensor shares the underlying buffer, so it remains valid
        only as long as the original response payload is alive.
        Zero-extent shapes return an empty CPU tensor instead.
        """

        if self.dtype != DTYPE:
            raise ProtocolError(f"tensor {self.name} is {self.dtype}, not {DTYPE}")
        try:
            import torch
        except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install bionemo-perception-client[torch] for torch tensor views") from error
        if self.data.nbytes == 0:
            return torch.empty(self.shape, dtype=torch.bfloat16, device="cpu")
        return torch.frombuffer(self.data, dtype=torch.bfloat16).reshape(self.shape)

    def torch_float32(self) -> Any:
        """Return an FP32 buffer view, or an empty CPU tensor for a zero-extent shape."""

        if self.dtype != DTYPE_F32:
            raise ProtocolError(f"tensor {self.name} is {self.dtype}, not {DTYPE_F32}")
        try:
            import torch
        except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install bionemo-perception-client[torch] for torch tensor views") from error
        if self.data.nbytes == 0:
            return torch.empty(self.shape, dtype=torch.float32, device="cpu")
        return torch.frombuffer(self.data, dtype=torch.float32).reshape(self.shape)


# Backward-compatible alias: the original transport was BF16-only.
BFloat16Tensor = WireTensor


@dataclass(frozen=True)
class ForwardResponse:
    """Parsed forward response."""

    metadata: Mapping[str, Any]
    payload_offset: int
    payload: memoryview
    tensors: tuple[WireTensor, ...]
    checksum_validated: bool
    tensor_metadata: tuple[Mapping[str, Any], ...]

    def tensor(self, name: str) -> WireTensor:
        """Return the uniquely named response tensor or raise KeyError."""
        matches = [tensor for tensor in self.tensors if tensor.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]


def parse_response(content: bytes | bytearray | memoryview, *, validate_checksum: bool = True) -> ForwardResponse:
    """Parse BNMOFWD1 or the BNMOFWD2 framing of older published images."""

    view = _byte_view(content)
    if len(view) < 8:
        raise ProtocolError("response envelope is truncated")
    if bytes(view[:8]) not in (FORWARD_MAGIC, LEGACY_V2_MAGIC):
        raise ProtocolError(f"response has unknown magic ({bytes(view[:8])!r})")
    return _parse_forward(view, validate_checksum=validate_checksum)


def parse_forward_response(
    content: bytes | bytearray | memoryview, *, validate_checksum: bool = True
) -> ForwardResponse:
    """Parse a forward response (BNMOFWD1)."""

    return parse_response(content, validate_checksum=validate_checksum)


def _parse_forward(view: memoryview, *, validate_checksum: bool) -> ForwardResponse:
    """Validate the envelope and construct tensor views and output metadata."""
    metadata, payload_offset, payload = _parse_envelope(view)
    expected_version = 2 if bytes(view[:8]) == LEGACY_V2_MAGIC else FORMAT_VERSION
    _validate_metadata(metadata, payload.nbytes, expected_version=expected_version)
    _validate_checksum(metadata, payload, validate_checksum)

    tensors = []
    tensor_meta = []
    for descriptor in metadata["tensors"]:
        start = descriptor["byte_offset"]
        end = start + descriptor["byte_length"]
        tensors.append(
            WireTensor(
                descriptor["name"],
                tuple(descriptor["shape"]),
                payload[start:end],
                metadata["dtype"],
            )
        )
        tensor_meta.append(
            {
                "name": descriptor["name"],
                "output_name": descriptor.get("output_name", ""),
                "resolved_layer": descriptor.get("resolved_layer", ""),
                "pooling": descriptor.get("pooling", {}),
                "row_offsets": descriptor.get("row_offsets", []),
            }
        )
    return ForwardResponse(
        metadata,
        payload_offset,
        payload,
        tuple(tensors),
        validate_checksum,
        tuple(tensor_meta),
    )


def _byte_view(content: bytes | bytearray | memoryview) -> memoryview:
    """Normalize a contiguous input buffer into an unsigned-byte view."""
    view = memoryview(content)
    if not view.contiguous:
        raise ProtocolError("response buffer must be contiguous")
    return view.cast("B")


def _parse_envelope(
    content: memoryview,
) -> tuple[dict[str, Any], int, memoryview]:
    """Decode framing and JSON metadata after validating offsets and padding."""
    if content.nbytes < PREAMBLE_BYTES:
        raise ProtocolError("response envelope is truncated")
    magic = bytes(content[:8])
    if magic not in (FORWARD_MAGIC, LEGACY_V2_MAGIC):
        raise ProtocolError("response envelope has invalid magic")
    version, reserved, header_length, payload_offset = struct.unpack_from("<HHIQ", content, 8)
    expected_version = 2 if magic == LEGACY_V2_MAGIC else FORMAT_VERSION
    if version != expected_version:
        raise ProtocolError(f"unsupported response envelope version {version}")
    if reserved != 0:
        raise ProtocolError("response envelope reserved bytes are non-zero")
    if payload_offset % PAYLOAD_ALIGNMENT:
        raise ProtocolError("response payload offset is not 64-byte aligned")
    header_end = PREAMBLE_BYTES + header_length
    if header_end > payload_offset or payload_offset > content.nbytes:
        raise ProtocolError("response envelope offsets are out of bounds")
    if any(content[header_end:payload_offset]):
        raise ProtocolError("response envelope alignment padding is non-zero")
    try:
        metadata = json.loads(bytes(content[PREAMBLE_BYTES:header_end]))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"response metadata is invalid JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise ProtocolError("response metadata must be a JSON object")
    return metadata, payload_offset, content[payload_offset:]


def _non_negative_int(value: object) -> bool:
    """Accept nonnegative integers while excluding booleans."""
    return type(value) is int and value >= 0


def _validate_metadata(metadata: dict[str, Any], payload_bytes: int, *, expected_version: int = FORMAT_VERSION) -> None:
    """Validate contract types, tensor descriptors, and contiguous payload geometry."""
    expected = {"format_version", "dtype", "tensors", "payload_bytes", "payload_checksum"}
    if set(metadata) != expected:
        raise ProtocolError("forward metadata fields are invalid")
    if not isinstance(metadata["dtype"], str) or type(metadata["format_version"]) is not int:
        raise ProtocolError("forward metadata contract fields are invalid")
    elt_size = DTYPE_ELT_SIZE.get(metadata["dtype"])
    if metadata["format_version"] != expected_version or elt_size is None:
        raise ProtocolError("forward metadata contract fields are invalid")
    if expected_version == 2 and metadata["dtype"] != DTYPE:
        raise ProtocolError("legacy BNMOFWD2 supports only bfloat16-le")
    tensors = metadata["tensors"]
    if not isinstance(tensors, list) or not tensors:
        raise ProtocolError("forward tensors must be a non-empty list")
    names: set[str] = set()
    expected_offset = 0
    for index, descriptor in enumerate(tensors):
        if not isinstance(descriptor, dict):
            raise ProtocolError(f"forward tensor {index} descriptor is invalid")
        required_keys = {"name", "output_name", "resolved_layer", "pooling", "shape", "byte_offset", "byte_length"}
        allowed_keys = required_keys | {"row_offsets"}
        if not required_keys.issubset(descriptor) or not set(descriptor).issubset(allowed_keys):
            raise ProtocolError(f"forward tensor {index} is missing required fields")
        name = descriptor["name"]
        shape = descriptor["shape"]
        if not isinstance(name, str) or not name or name in names:
            raise ProtocolError("forward tensor names must be non-empty and unique")
        names.add(name)
        # Zero-extent axes are valid (e.g. KERMT bond views for a bond-less
        # molecule); such tensors carry zero payload bytes.
        if not isinstance(shape, list) or not shape or not all(_non_negative_int(d) for d in shape):
            raise ProtocolError(f"forward tensor {name} shape is invalid")
        byte_length = elt_size
        for d in shape:
            byte_length *= d
        if (
            not _non_negative_int(descriptor["byte_offset"])
            or not _non_negative_int(descriptor["byte_length"])
            or descriptor["byte_offset"] != expected_offset
            or descriptor["byte_length"] != byte_length
        ):
            raise ProtocolError(f"forward tensor {name} byte geometry is invalid")
        expected_offset += byte_length
    if (
        not _non_negative_int(metadata["payload_bytes"])
        or metadata["payload_bytes"] != expected_offset
        or payload_bytes != expected_offset
    ):
        raise ProtocolError("forward payload length differs from metadata")


def _validate_checksum(metadata: dict[str, Any], payload: memoryview, validate: bool) -> None:
    """Check the checksum descriptor and optionally verify payload integrity."""
    checksum = metadata.get("payload_checksum")
    if (
        not isinstance(checksum, dict)
        or set(checksum) != {"algorithm", "digest"}
        or checksum.get("algorithm") != "xxh64"
        or not isinstance(checksum.get("digest"), str)
        or len(checksum["digest"]) != 16
        or any(c not in "0123456789abcdef" for c in checksum["digest"])
    ):
        raise ProtocolError("payload checksum descriptor is invalid")
    if validate and checksum["digest"] != f"{_xxh64(payload):016x}":
        raise ProtocolError("payload checksum mismatch")


def _xxh64(content: bytes | bytearray | memoryview, seed: int = 0) -> int:
    """Return canonical XXH64 using the native one-shot implementation."""

    return xxhash.xxh64_intdigest(_byte_view(content), seed=seed)
