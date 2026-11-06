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

"""Small typed HTTP clients for BioNeMo Perception Container forward endpoints."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar
from urllib.parse import urlsplit

from typing_extensions import NotRequired, TypedDict, Unpack

from .protocol import CONTENT_TYPE, ForwardResponse, parse_response


class PoolingSpec(TypedDict):
    """Reduction applied independently to each requested output."""

    kind: Literal["none", "mean"]
    window: NotRequired[int]


class OutputSpec(TypedDict):
    """One named model capture in the canonical forward schema."""

    layer: str
    name: NotRequired[str]
    pooling: NotRequired[PoolingSpec]


class ForwardRequest(TypedDict):
    """Fields shared by Evo 2 and ESM-3 forward requests."""

    sequence: str | list[str]
    outputs: list[OutputSpec]


class Evo2ForwardRequest(ForwardRequest):
    """Canonical Evo 2 forward request."""


StructureCoordinates = list[list[list[float]]] | list[list[list[list[float]]]]
ScalarTrack = list[int] | list[list[int]]
ConfidenceTrack = list[float] | list[list[float]]
VectorTrack = list[list[int]] | list[list[list[int]]]


class ESM3ForwardRequest(ForwardRequest):
    """Canonical ESM-3 forward request, including its optional model tracks."""

    structure_coords: NotRequired[StructureCoordinates]
    per_res_plddt: NotRequired[ConfidenceTrack]
    structure_tokens: NotRequired[ScalarTrack]
    ss8_tokens: NotRequired[ScalarTrack]
    sasa_tokens: NotRequired[ScalarTrack]
    function_tokens: NotRequired[VectorTrack]
    residue_annotation_tokens: NotRequired[VectorTrack]


class KermtForwardRequest(TypedDict):
    """Canonical KERMT forward request (SMILES in place of ``sequence``).

    Supported output layers: ``atom_from_atom``, ``atom_from_bond``,
    ``bond_from_atom``, ``bond_from_bond``, ``node_mt_out``, ``atom_mean``,
    ``atom_concat``, and the molecule-level ``molecule_fingerprint``.
    Pooling (``mean``/windowed ``mean``) applies per molecule over the
    layer's atom or bond rows.
    """

    smiles: str | list[str]
    outputs: list[OutputSpec]


class PerceptionRequestError(RuntimeError):
    """An inference container returned a non-successful HTTP response."""

    def __init__(self, status_code: int, body: str, url: str) -> None:
        self.status_code = status_code
        self.body = body
        self.url = url
        detail = body or "empty response body"
        super().__init__(f"Request to {url} failed with HTTP {status_code}: {detail}")


RequestT = TypeVar("RequestT")


@dataclass(frozen=True, slots=True)
class PerceptionClient(Generic[RequestT]):
    """Shared transport for a model-specific, typed BioNeMo Perception Container client.

    ``token`` is the bare access token for token-gated endpoints (for example
    Lepton endpoints created with ``--tokens``); it is sent as
    ``Authorization: Bearer <token>``. Leave it unset for open endpoints.
    """

    url: str
    timeout: float = 120.0
    token: str | None = None

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be an absolute HTTP(S) URL")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.token is not None:
            normalized = self.token.strip()
            if not normalized:
                raise ValueError("token must be non-empty")
            if any(character.isspace() for character in normalized):
                raise ValueError("token must not contain whitespace")
            object.__setattr__(self, "token", normalized)

    @property
    def forward_url(self) -> str:
        """Return the canonical endpoint, accepting either an origin or endpoint URL."""

        url = self.url.rstrip("/")
        return url if url.endswith("/v1/forward") else f"{url}/v1/forward"

    def forward_payload(self, request: RequestT) -> ForwardResponse:
        """Send an already assembled, model-specific forward request."""

        body = json.dumps(request, separators=(",", ":")).encode()
        request_url = self.forward_url
        headers = {
            "Accept": CONTENT_TYPE,
            "Content-Type": "application/json",
            "User-Agent": "bionemo-perception-client/0.1",
        }
        if self.token is not None:
            headers["Authorization"] = f"Bearer {self.token}"
        http_request = urllib.request.Request(
            request_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                return parse_response(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace").strip()
            raise PerceptionRequestError(error.code, detail, request_url) from error


@dataclass(frozen=True, slots=True)
class ESM3Client(PerceptionClient[ESM3ForwardRequest]):
    """Typed client for ESM-3's canonical ``/v1/forward`` request."""

    def forward(self, **kwargs: Unpack[ESM3ForwardRequest]) -> ForwardResponse:
        return self.forward_payload(kwargs)


@dataclass(frozen=True, slots=True)
class Evo2Client(PerceptionClient[Evo2ForwardRequest]):
    """Typed client for Evo 2's canonical ``/v1/forward`` request."""

    def forward(self, **kwargs: Unpack[Evo2ForwardRequest]) -> ForwardResponse:
        return self.forward_payload(kwargs)


@dataclass(frozen=True, slots=True)
class KERMTClient(PerceptionClient[KermtForwardRequest]):
    """Typed client for KERMT's canonical ``/v1/forward`` request."""

    def forward(self, **kwargs: Unpack[KermtForwardRequest]) -> ForwardResponse:
        return self.forward_payload(kwargs)
