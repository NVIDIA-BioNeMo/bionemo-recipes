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

"""Recipe-local skill packaging and discovery without repository dependencies."""

from pathlib import Path

import yaml


def test_skill_discovery():
    recipe = Path(__file__).resolve().parents[1]
    discovery = recipe / ".agents" / "skills"
    assert discovery.is_symlink()
    assert discovery.resolve() == recipe / "skills"
    for skill in discovery.iterdir():
        entry = skill / "SKILL.md"
        frontmatter = entry.read_text().split("---", 2)[1]
        metadata = yaml.safe_load(frontmatter)
        assert metadata["name"] == skill.name
        assert metadata["description"]
        assert entry.resolve().is_relative_to(recipe)
