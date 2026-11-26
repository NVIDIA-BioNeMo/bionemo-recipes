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

from codecs import BOM_UTF8

import pytest

from ci.scripts.license_check import process_file


@pytest.mark.parametrize(
    "preamble",
    [
        b"# coding: latin-1\n",
        b"#!/usr/bin/env python3\n# coding: latin-1\n",
        b"# Editor settings\n# coding=latin-1\n",
        b"\n# -*- coding: latin-1 -*-\n",
    ],
)
def test_preserves_python_encoding_cookie(tmp_path, preamble):
    path = tmp_path / "source.py"
    body = b"name = 'caf\xe9'\n"
    path.write_bytes(preamble + body)

    process_file(path, dry_run=False)

    updated = path.read_bytes()
    assert updated.startswith(preamble)
    assert updated.endswith(body)
    assert b"SPDX-License-Identifier: LicenseRef-Apache2" in updated
    namespace = {}
    exec(compile(updated, str(path), "exec"), namespace)
    assert namespace["name"] == "café"


def test_keeps_utf8_bom_and_line_endings(tmp_path):
    path = tmp_path / "source.py"
    body = "name = 'café'".encode()
    path.write_bytes(BOM_UTF8 + b"# coding: utf-8\r\n" + body)

    process_file(path, dry_run=False)

    updated = path.read_bytes()
    assert updated.startswith(BOM_UTF8 + b"# coding: utf-8\r\n\r\n# SPDX-FileCopyrightText:")
    assert b"\n" not in updated.replace(b"\r\n", b"")
    assert updated.endswith(body)
    compile(updated, str(path), "exec")


@pytest.mark.parametrize("line_end", [b"", b"\n", b"\r\n"])
def test_shebang_only_file(tmp_path, line_end):
    path = tmp_path / "source.py"
    shebang = b"#!/usr/bin/env python3"
    path.write_bytes(shebang + line_end)

    process_file(path, dry_run=False)

    newline = line_end or b"\n"
    assert path.read_bytes().startswith(shebang + newline * 2 + b"# SPDX-FileCopyrightText:")


@pytest.mark.parametrize(("suffix", "body"), [(".py", b"pass"), (".rs", b"fn main() {}")])
def test_preserves_mixed_line_endings_and_is_idempotent(tmp_path, suffix, body):
    path = tmp_path / f"source{suffix}"
    body = body + b"\r\n\n"
    path.write_bytes(body)

    process_file(path, dry_run=False)
    updated = path.read_bytes()
    process_file(path, dry_run=False)

    comment = b"#" if suffix == ".py" else b"//"
    assert updated.endswith(body)
    assert updated.startswith(comment + b" SPDX-FileCopyrightText:")
    assert b"\r\n" + comment + b" SPDX-License-Identifier:" in updated
    assert path.read_bytes() == updated


def test_dry_run_and_opt_out_keep_source_bytes(tmp_path):
    path = tmp_path / "source.py"
    original = b"# coding: latin-1\nname = 'caf\xe9'\n"
    path.write_bytes(original)

    process_file(path, dry_run=True)
    assert path.read_bytes() == original

    original = b"# coding: latin-1\n\n# noqa: license-check\nname = 'caf\xe9'\n"
    path.write_bytes(original)

    process_file(path, dry_run=False)
    assert path.read_bytes() == original
