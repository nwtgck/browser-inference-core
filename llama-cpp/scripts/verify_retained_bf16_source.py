#!/usr/bin/env python3
"""Reject unreviewed source contracts before enabling experimental BF16 retention."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]


def verify(source: Path, contract: Path = ROOT / "config/retained-bf16-source.json") -> None:
    expected = json.loads(contract.read_text())
    files = expected.get("files") if isinstance(expected, dict) else None
    if not isinstance(files, dict) or not files:
        raise ValueError("retain-checked BF16 needs semantic review: empty/invalid source contract")
    for relative, digest in files.items():
        name = PurePosixPath(relative)
        if (not relative or name.is_absolute() or ".." in name.parts or "\\" in relative or
                not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("retain-checked BF16 needs semantic review: invalid contract entry")
        path = source / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("retain-checked BF16 needs semantic review: " + relative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    verify(args.source)


if __name__ == "__main__":
    main()
