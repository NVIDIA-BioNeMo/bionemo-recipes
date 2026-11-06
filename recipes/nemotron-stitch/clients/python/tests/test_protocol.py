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

from __future__ import annotations

import json
import struct
import unittest
from importlib.util import find_spec

from bionemo_perception import (
    CONTENT_TYPE,
    ForwardResponse,
    ProtocolError,
    parse_forward_response,
    parse_response,
)
from bionemo_perception.protocol import FORWARD_MAGIC, _xxh64


def envelope(payload: bytes, tensors: list[dict], dtype: str = "bfloat16-le") -> bytes:
    """Build a valid BNMOFWD1 envelope."""

    metadata: dict[str, object] = {
        "format_version": 1,
        "dtype": dtype,
        "tensors": tensors,
        "payload_bytes": len(payload),
    }
    metadata["payload_checksum"] = {
        "algorithm": "xxh64",
        "digest": f"{_xxh64(payload):016x}",
    }
    header = json.dumps(metadata, separators=(",", ":")).encode()
    payload_offset = (24 + len(header) + 63) // 64 * 64
    return b"".join(
        (
            FORWARD_MAGIC,
            struct.pack("<HHIQ", 1, 0, len(header), payload_offset),
            header,
            bytes(payload_offset - 24 - len(header)),
            payload,
        )
    )


class TestParser(unittest.TestCase):
    @unittest.skipUnless(find_spec("numpy"), "NumPy features require the numpy extra")
    def test_numpy_features(self) -> None:
        import numpy as np

        for dtype, payload in (
            ("bfloat16-le", struct.pack("<4H", 0x3F80, 0x4000, 0x4040, 0x4080)),
            ("float32-le", struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)),
        ):
            with self.subTest(dtype=dtype):
                body = envelope(
                    payload,
                    [
                        {
                            "name": "features.output",
                            "output_name": "features",
                            "resolved_layer": "encoder",
                            "pooling": {"kind": "none"},
                            "shape": [2, 2],
                            "byte_offset": 0,
                            "byte_length": len(payload),
                        }
                    ],
                    dtype=dtype,
                )
                features = parse_response(body).tensor("features.output").numpy_float32()
                self.assertEqual(features.dtype, np.dtype("float32"))
                np.testing.assert_array_equal(features, [[1.0, 2.0], [3.0, 4.0]])

    def test_legacy_framing(self) -> None:
        payload = struct.pack("<H", 0x3F80)
        body = envelope(
            payload,
            [
                {
                    "name": "features.output",
                    "output_name": "features",
                    "resolved_layer": "encoder",
                    "pooling": {"kind": "none"},
                    "shape": [1],
                    "byte_offset": 0,
                    "byte_length": len(payload),
                }
            ],
            dtype="bfloat16-le",
        )
        view = bytearray(body)
        view[:8] = b"BNMOFWD2"
        struct.pack_into("<H", view, 8, 2)
        view[24:] = view[24:].replace(b'"format_version":1', b'"format_version":2', 1)
        response = parse_response(view)
        self.assertTrue(response.checksum_validated)
        self.assertEqual(response.tensor("features.output").shape, (1,))

    @unittest.skipUnless(find_spec("torch"), "torch tensor views require the torch extra")
    def test_torch_features(self) -> None:
        import torch

        for dtype, payload, torch_dtype in (
            ("bfloat16-le", struct.pack("<4H", 0x3F80, 0x4000, 0x4040, 0x4080), torch.bfloat16),
            ("float32-le", struct.pack("<4f", 1.0, 2.0, 3.0, 4.0), torch.float32),
        ):
            with self.subTest(dtype=dtype):
                # A writable buffer also exercises the zero-copy view contract.
                body = bytearray(
                    envelope(
                        payload,
                        [
                            {
                                "name": "features.output",
                                "output_name": "features",
                                "resolved_layer": "encoder",
                                "pooling": {"kind": "none"},
                                "shape": [2, 2],
                                "byte_offset": 0,
                                "byte_length": len(payload),
                            }
                        ],
                        dtype=dtype,
                    )
                )
                response = parse_response(body)
                wire = response.tensor("features.output")
                convert = wire.torch_bfloat16 if dtype == "bfloat16-le" else wire.torch_float32
                view = convert()
                self.assertEqual(view.dtype, torch_dtype)
                features = view.clone()
                torch.testing.assert_close(features.float(), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
                view[0, 0] = 5
                self.assertEqual(convert()[0, 0].item(), 5)
                self.assertEqual(features[0, 0].item(), 1)
                wrong_convert = wire.torch_float32 if dtype == "bfloat16-le" else wire.torch_bfloat16
                with self.assertRaises(ProtocolError):
                    wrong_convert()

    def test_float32_round_trip(self) -> None:
        payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        body = envelope(
            payload,
            [
                {
                    "name": "sequence_0.kermt.output",
                    "output_name": "kermt",
                    "resolved_layer": "atom_concat",
                    "pooling": {"kind": "none"},
                    "shape": [4],
                    "byte_offset": 0,
                    "byte_length": 16,
                },
            ],
            dtype="float32-le",
        )
        response = parse_forward_response(body)
        tensor = response.tensor("sequence_0.kermt.output")
        self.assertEqual(tensor.dtype, "float32-le")
        self.assertEqual(struct.unpack("<4f", tensor.data), (1.0, 2.0, 3.0, 4.0))

    def test_content_type(self) -> None:
        self.assertEqual(CONTENT_TYPE, "application/vnd.bnmo.tensor+binary; version=1")

    def test_round_trip(self) -> None:
        payload = struct.pack("<4H", 0x3F80, 0x4000, 0x4040, 0x4080)
        body = envelope(
            payload,
            [
                {
                    "name": "decoder.layers.20.output",
                    "output_name": "decoder.layers.20",
                    "resolved_layer": "decoder.layers.20",
                    "pooling": {"kind": "none"},
                    "shape": [4],
                    "byte_offset": 0,
                    "byte_length": 8,
                },
            ],
        )
        response = parse_forward_response(body)
        self.assertIsInstance(response, ForwardResponse)
        self.assertEqual(len(response.tensors), 1)
        self.assertEqual(response.tensors[0].name, "decoder.layers.20.output")
        self.assertEqual(response.tensors[0].shape, (4,))
        self.assertEqual(response.tensors[0].data.nbytes, 8)
        self.assertEqual(len(response.tensor_metadata), 1)
        self.assertEqual(response.tensor_metadata[0]["output_name"], "decoder.layers.20")
        self.assertEqual(response.tensor_metadata[0]["pooling"], {"kind": "none"})
        self.assertTrue(response.checksum_validated)

    def test_with_pooling_and_row_offsets(self) -> None:
        payload = struct.pack("<2H", 0, 1)
        body = envelope(
            payload,
            [
                {
                    "name": "mean_20.output",
                    "output_name": "mean_20",
                    "resolved_layer": "decoder.layers.20",
                    "pooling": {"kind": "mean", "window": 64},
                    "shape": [2],
                    "byte_offset": 0,
                    "byte_length": 4,
                    "row_offsets": [0, 1],
                },
            ],
        )
        response = parse_forward_response(body)
        self.assertEqual(response.tensor_metadata[0]["row_offsets"], [0, 1])
        self.assertEqual(response.tensor_metadata[0]["pooling"], {"kind": "mean", "window": 64})

    def test_checksum_validation_can_be_skipped(self) -> None:
        payload = struct.pack("<2H", 0, 1)
        body = bytearray(
            envelope(
                payload,
                [
                    {
                        "name": "x.output",
                        "output_name": "x",
                        "resolved_layer": "x",
                        "pooling": {"kind": "none"},
                        "shape": [2],
                        "byte_offset": 0,
                        "byte_length": 4,
                    },
                ],
            )
        )
        body[-1] ^= 1
        response = parse_forward_response(bytes(body), validate_checksum=False)
        self.assertFalse(response.checksum_validated)

    def test_rejects_corruption(self) -> None:
        payload = struct.pack("<2H", 0, 1)
        body = bytearray(
            envelope(
                payload,
                [
                    {
                        "name": "x.output",
                        "output_name": "x",
                        "resolved_layer": "x",
                        "pooling": {"kind": "none"},
                        "shape": [2],
                        "byte_offset": 0,
                        "byte_length": 4,
                    },
                ],
            )
        )
        body[-1] ^= 1
        with self.assertRaises(ProtocolError):
            parse_forward_response(bytes(body))

    def test_unknown_magic_rejected(self) -> None:
        bad = b"UNKNOWN0" + b"\x00" * 16
        with self.assertRaises(ProtocolError):
            parse_response(bad)

    def test_truncated_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_response(b"\x00" * 4)

    def test_parse_response_and_parse_forward_response_are_synonyms(self) -> None:
        payload = struct.pack("<2H", 0, 1)
        body = envelope(
            payload,
            [
                {
                    "name": "x.output",
                    "output_name": "x",
                    "resolved_layer": "x",
                    "pooling": {"kind": "none"},
                    "shape": [2],
                    "byte_offset": 0,
                    "byte_length": 4,
                },
            ],
        )
        r1 = parse_response(body)
        r2 = parse_forward_response(body)
        self.assertEqual(r1.tensors[0].name, r2.tensors[0].name)

    def test_named_lookup(self) -> None:
        payload = struct.pack("<4H", 0, 1, 2, 3)
        body = envelope(
            payload,
            [
                {
                    "name": "a.output",
                    "output_name": "a",
                    "resolved_layer": "a",
                    "pooling": {"kind": "none"},
                    "shape": [2],
                    "byte_offset": 0,
                    "byte_length": 4,
                },
                {
                    "name": "b.output",
                    "output_name": "b",
                    "resolved_layer": "b",
                    "pooling": {"kind": "none"},
                    "shape": [2],
                    "byte_offset": 4,
                    "byte_length": 4,
                },
            ],
        )
        response = parse_forward_response(body)
        self.assertEqual(response.tensor("a.output").shape, (2,))
        self.assertEqual(response.tensor("b.output").shape, (2,))
        with self.assertRaises(KeyError):
            response.tensor("missing")


if __name__ == "__main__":
    unittest.main()
