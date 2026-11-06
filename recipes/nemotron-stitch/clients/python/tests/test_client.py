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

"""HTTP transport and authentication regression tests."""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock, patch

from bionemo_perception import (
    CONTENT_TYPE,
    ESM3Client,
    Evo2Client,
    ForwardResponse,
    KERMTClient,
    PerceptionRequestError,
)


class StubResponse:
    """In-memory HTTP response used by transport tests."""

    def __init__(self, body: bytes) -> None:
        """Store a synthetic HTTP response body."""
        self.body = body

    def __enter__(self) -> StubResponse:
        """Expose the response body through the context-manager protocol."""
        return self

    def __exit__(self, *_: object) -> None:
        """Finish the in-memory response context without suppressing exceptions."""
        return None

    def read(self) -> bytes:
        """Return the synthetic response body."""
        return self.body


class TestTypedClients(unittest.TestCase):
    """Request serialization, authentication, and error-handling contracts."""

    @patch("bionemo_perception.client.parse_response")
    @patch("bionemo_perception.client.urllib.request.urlopen")
    def test_kermt_smiles(self, urlopen: Mock, parse: Mock) -> None:
        """KERMT serializes SMILES inputs without a sequence field."""
        parse.return_value = Mock(spec=ForwardResponse)
        urlopen.return_value = StubResponse(b"tensor envelope")
        client = KERMTClient("https://encoder.example")
        actual = client.forward(
            smiles=["CCO", "C"],
            outputs=[{"layer": "atom_concat", "name": "features", "pooling": {"kind": "none"}}],
        )
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data)["smiles"], ["CCO", "C"])
        self.assertNotIn("sequence", json.loads(request.data))
        self.assertIs(actual, parse.return_value)

    @patch("bionemo_perception.client.parse_response")
    @patch("bionemo_perception.client.urllib.request.urlopen")
    def test_esm3_serializes_typed_tracks(self, urlopen: Mock, parse: Mock) -> None:
        """ESM-3 includes auxiliary tracks and binary-response negotiation."""
        expected = Mock(spec=ForwardResponse)
        parse.return_value = expected
        urlopen.return_value = StubResponse(b"tensor envelope")

        client = ESM3Client("http://localhost:8000")
        actual = client.forward(
            sequence="AC",
            structure_coords=[
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            ],
            per_res_plddt=[0.25, 0.75],
            outputs=[{"layer": "output_layer"}],
        )

        self.assertIs(actual, expected)
        http_request = urlopen.call_args.args[0]
        self.assertEqual(http_request.full_url, "http://localhost:8000/v1/forward")
        self.assertEqual(http_request.get_header("Accept"), CONTENT_TYPE)
        self.assertEqual(http_request.get_header("Content-type"), "application/json")
        payload = json.loads(http_request.data)
        self.assertEqual(payload["sequence"], "AC")
        self.assertEqual(payload["per_res_plddt"], [0.25, 0.75])
        parse.assert_called_once_with(b"tensor envelope")

    @patch("bionemo_perception.client.parse_response")
    @patch("bionemo_perception.client.urllib.request.urlopen")
    def test_evo2_accepts_full_endpoint_url(self, urlopen: Mock, parse: Mock) -> None:
        """A full forward URL is preserved along with the requested timeout."""
        parse.return_value = Mock(spec=ForwardResponse)
        urlopen.return_value = StubResponse(b"tensor envelope")

        client = Evo2Client("https://ic.example/v1/forward", timeout=5)
        client.forward(sequence=["ACGT", "TGCA"], outputs=[{"layer": "output_layer"}])

        self.assertEqual(urlopen.call_args.args[0].full_url, "https://ic.example/v1/forward")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 5)

    @patch("bionemo_perception.client.parse_response")
    @patch("bionemo_perception.client.urllib.request.urlopen")
    def test_token_adds_bearer_authorization(self, urlopen: Mock, parse: Mock) -> None:
        """The initial request carries the configured bearer token."""
        parse.return_value = Mock(spec=ForwardResponse)
        urlopen.return_value = StubResponse(b"tensor envelope")

        client = Evo2Client("https://ic.example", token="test-token")
        client.forward(sequence="ACGT", outputs=[{"layer": "output_layer"}])

        http_request = urlopen.call_args.args[0]
        self.assertEqual(http_request.get_header("Authorization"), "Bearer test-token")

    @patch("bionemo_perception.client.parse_response")
    @patch("bionemo_perception.client.urllib.request.urlopen")
    def test_redirect_auth(self, urlopen: Mock, parse: Mock) -> None:
        """The stdlib redirect handler must never copy the original bearer token."""
        client = Evo2Client("https://encoder.example", token="test-token")
        parse.return_value = Mock(spec=ForwardResponse)
        urlopen.return_value = StubResponse(b"tensor envelope")
        client.forward(sequence="ACGT", outputs=[{"layer": "output_layer"}])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        for status in (301, 302, 303):
            for destination in (
                "https://other.example/v1/forward",
                "http://encoder.example/v1/forward",
                "https://encoder.example:8443/v1/forward",
                "https://encoder.example/redirected",
            ):
                with self.subTest(status=status, destination=destination):
                    redirected = urllib.request.HTTPRedirectHandler().redirect_request(
                        request, None, status, "redirect", {}, destination
                    )
                    self.assertIsNotNone(redirected)
                    self.assertIsNone(redirected.get_header("Authorization"))
                    self.assertEqual(redirected.get_header("Accept"), CONTENT_TYPE)

    @patch("bionemo_perception.client.parse_response")
    @patch("bionemo_perception.client.urllib.request.urlopen")
    def test_no_token_omits_authorization(self, urlopen: Mock, parse: Mock) -> None:
        """Open services receive no authorization header."""
        parse.return_value = Mock(spec=ForwardResponse)
        urlopen.return_value = StubResponse(b"tensor envelope")

        ESM3Client("https://ic.example").forward(sequence="AC", outputs=[{"layer": "output_layer"}])

        http_request = urlopen.call_args.args[0]
        self.assertIsNone(http_request.get_header("Authorization"))

    @patch("bionemo_perception.client.parse_response")
    @patch("bionemo_perception.client.urllib.request.urlopen")
    def test_token_surrounding_whitespace_is_stripped(self, urlopen: Mock, parse: Mock) -> None:
        """Tokens read from files tolerate surrounding whitespace."""
        parse.return_value = Mock(spec=ForwardResponse)
        urlopen.return_value = StubResponse(b"tensor envelope")

        client = Evo2Client("https://ic.example", token="  test-token\n")
        client.forward(sequence="ACGT", outputs=[{"layer": "output_layer"}])

        http_request = urlopen.call_args.args[0]
        self.assertEqual(http_request.get_header("Authorization"), "Bearer test-token")

    def test_token_validation(self) -> None:
        """Empty tokens and internal whitespace fail during construction."""
        with self.assertRaisesRegex(ValueError, "token must be non-empty"):
            Evo2Client("http://localhost:8000", token="   ")
        with self.assertRaisesRegex(ValueError, "token must not contain whitespace"):
            ESM3Client("http://localhost:8000", token="test token")

    @patch("bionemo_perception.client.urllib.request.urlopen")
    def test_http_error_includes_response_body(self, urlopen: Mock) -> None:
        """HTTP failures retain the response detail and close its stream."""
        body = io.BytesIO(b'{"error":"unknown_layer"}')
        urlopen.side_effect = urllib.error.HTTPError(
            "http://localhost:8000/v1/forward",
            422,
            "Unprocessable Entity",
            {},
            body,
        )

        with self.assertRaisesRegex(PerceptionRequestError, "unknown_layer") as raised:
            Evo2Client("http://localhost:8000").forward(sequence="ACGT", outputs=[{"layer": "missing"}])
        self.assertEqual(raised.exception.status_code, 422)
        self.assertTrue(body.closed)

    @patch("bionemo_perception.client.urllib.request.urlopen")
    def test_error_read_timeout(self, urlopen: Mock) -> None:
        """An interrupted error-body read closes the response and preserves the exception."""
        body = Mock(spec=io.BytesIO)
        body.closed = False
        failure = TimeoutError("error body timed out")
        body.read.side_effect = failure
        error = urllib.error.HTTPError("https://encoder.example/v1/forward", 503, "unavailable", {}, body)
        urlopen.side_effect = error
        with self.assertRaises(TimeoutError) as raised:
            Evo2Client("https://encoder.example").forward(sequence="ACGT", outputs=[{"layer": "output_layer"}])
        self.assertIs(raised.exception, failure)
        body.close.assert_called_once()

    def test_configuration_validation(self) -> None:
        """Relative URLs and nonpositive timeouts fail during construction."""
        with self.assertRaisesRegex(ValueError, "absolute HTTP"):
            ESM3Client("localhost:8000")
        with self.assertRaisesRegex(ValueError, "timeout must be positive"):
            Evo2Client("http://localhost:8000", timeout=0)


if __name__ == "__main__":
    unittest.main()
