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

"""Static contract probes for the model-specific forward keyword schemas."""

from bionemo_perception import (
    ESM3Client,
    Evo2Client,
    Evo2ForwardRequest,
    PerceptionClient,
)


def check_forward_signatures() -> None:
    esm3 = ESM3Client("http://localhost:8000")
    esm3.forward(
        sequence="AC",
        structure_coords=[
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ],
        per_res_plddt=[0.25, 0.75],
        outputs=[{"layer": "output_layer"}],
    )

    evo2 = Evo2Client("http://localhost:8000")
    evo2.forward(sequence="ACGT", outputs=[{"layer": "output_layer"}])

    authenticated = Evo2Client("https://ic.example", timeout=30, token="tk")
    authenticated.forward(sequence="ACGT", outputs=[{"layer": "output_layer"}])

    generic = PerceptionClient[Evo2ForwardRequest]("http://localhost:8000")
    generic.forward_payload({"sequence": "ACGT", "outputs": [{"layer": "output_layer"}]})

    evo2.forward(  # type: ignore[call-arg]
        sequence="ACGT",
        outputs=[{"layer": "output_layer"}],
        structure_coords=[],  # pyright: ignore[reportCallIssue]
    )
