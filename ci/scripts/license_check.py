#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import logging
import re
import textwrap
from codecs import BOM_UTF8
from datetime import datetime
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

year = datetime.now().year

license_text = textwrap.dedent("""\
    SPDX-License-Identifier: LicenseRef-Apache2

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """)

default_copyright_text = (
    f"SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved."
)
default_combined_license = "\n".join([default_copyright_text, license_text])

copyright_regex_pattern = (
    r"SPDX-FileCopyrightText: Copyright \(c\) \d{4} NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.\n?\r?"
    r"(?:SPDX-FileCopyrightText: Copyright \(c\) \d{4}.*\n?\r?)*"
)
license_regex_pattern = copyright_regex_pattern + re.escape(license_text)
license_regex = re.compile(license_regex_pattern)

# PEP 263 permits a coding cookie on the first line, or on the second line
# when the first line contains only a comment or whitespace.
_PYTHON_CODING_COOKIE = re.compile(rb"^[ \t\f]*#.*?coding[:=][ \t]*[-_.a-zA-Z0-9]+")
_PYTHON_COOKIE_PREFIX = re.compile(rb"^[ \t\f]*(?:[#\r\n]|$)")

# Basic regex sanity checks.
assert re.compile(copyright_regex_pattern).match(default_copyright_text), "Default copyright text not valid"
assert license_regex.match(default_combined_license), "Default license text or regex is not valid"


def _preamble_length(lines: list[bytes], *, python_source: bool) -> int:
    """Count the initial lines that must stay ahead of an inserted header."""
    if not lines:
        return 0

    first = lines[0]
    if python_source:
        if _PYTHON_CODING_COOKIE.match(first):
            return 1
        if len(lines) > 1 and _PYTHON_COOKIE_PREFIX.match(first) and _PYTHON_CODING_COOKIE.match(lines[1]):
            return 2

    return int(first.startswith(b"#!"))


def process_file(filepath: Path, dry_run: bool):
    """Process a file to ensure it has a valid license block, or add a new one if it doesn't."""
    comment_start = get_comment_delimiter(filepath)
    comment_bytes = comment_start.encode("ascii")
    source = filepath.read_bytes()
    bom = BOM_UTF8 if source.startswith(BOM_UTF8) else b""
    lines = source[len(bom) :].splitlines(keepends=True)

    start_line = _preamble_length(lines, python_source=filepath.suffix == ".py")
    needs_separator = start_line > 0
    if needs_separator and start_line < len(lines) and not lines[start_line].strip():
        start_line += 1
        needs_separator = False

    license_block = []
    for line in lines[start_line:]:
        if not line.startswith(comment_bytes):
            break
        # The required header is ASCII. Inspect it without decoding the source.
        license_block.append(line.rstrip(b"\r\n").decode("ascii", errors="replace"))

    def uncomment(text: str) -> str:
        return re.sub(rf"^{comment_start}\ ?", "", text)

    if "# noqa: license-check" in license_block:
        logger.info(f"Skipping {filepath} because it contains `# noqa: license-check`.")
        return

    license_block_text = "\n".join(uncomment(line) for line in license_block) + "\n"
    if len(license_block) != 0 and license_regex.match(license_block_text):
        logger.info(f"Skipping {filepath} because it contains a valid license block.")
        return

    logger.info(f"Adding license block to {filepath}.")
    if dry_run:
        return

    first_line_break = re.search(rb"\r\n?|\n", source)
    newline = first_line_break[0] if first_line_break else b"\n"
    preamble = bom + b"".join(lines[:start_line])
    if start_line and not preamble.endswith((b"\r", b"\n")):
        preamble += newline
    if needs_separator:
        preamble += newline

    header = newline.join(
        f"{comment_start} {line}".rstrip().encode("ascii") for line in default_combined_license.splitlines()
    )
    filepath.write_bytes(preamble + header + newline * 2 + b"".join(lines[start_line:]))


def get_comment_delimiter(filepath: Path) -> str:
    """Get the comment delimiter for a file based on its extension."""
    match filepath.suffix:
        case ".py":
            return "#"
        case ".rs":
            return "//"
        case _:
            raise ValueError(f"Unsupported file type: {filepath}")


def main():
    """Main entry point for the license check script."""
    parser = argparse.ArgumentParser(description="Ensure files have proper license headers")
    parser.add_argument("files", nargs="+", help="Files to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")

    args = parser.parse_args()

    for filename in args.files:
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filename} does not exist")

        if filepath.suffix not in [".py", ".rs"]:
            raise ValueError(f"Unsupported file type: {filepath}")

        process_file(filepath, args.dry_run)


if __name__ == "__main__":
    main()
