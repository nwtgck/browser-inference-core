#!/usr/bin/env python3
"""Annotate the generated sibling Wasm URL without changing runtime behavior."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

SCRIPT_PATH = 'scripts/prepare_browser_module.py'
ANNOTATION = b'/* @vite-ignore */'
WASM_URL = re.compile(rb'\bnew\s+URL\s*\(\s*(?:/\*[\s\S]*?\*/\s*)*[\'"`][^\'"`\r\n]*\.wasm[^\'"`\r\n]*[\'"`]')
CORE_URL = re.compile(rb'\bnew\s+URL\s*\(\s*(?P<annotation>/\* @vite-ignore \*/\s*)?'
                      rb'(?P<argument>(?P<quote>[\'"])core\.wasm(?P=quote))\s*,\s*'
                      rb'import\s*\.\s*meta\s*\.\s*url\s*,?\s*\)')


def identity(source: bytes) -> dict:
    return {'bytes': len(source), 'sha256': hashlib.sha256(source).hexdigest()}


def annotate(source: bytes) -> bytes:
    matches = list(CORE_URL.finditer(source))
    if len(matches) != 1 or len(WASM_URL.findall(source)) != 1:
        raise ValueError('Unknown generated browser Wasm URL boundary; expected one sibling core.wasm URL')
    match = matches[0]
    if match.group('annotation') is not None:
        return source
    position = match.start('argument')
    return source[:position] + ANNOTATION + source[position:]


def prepare(module: Path) -> dict:
    if module.is_symlink() or not module.is_file():
        raise ValueError(f'Missing or linked generated browser module: {module}')
    source = module.read_bytes()
    result = annotate(source)
    implementation = Path(__file__)
    record = {'implementation': {'path': SCRIPT_PATH, **identity(implementation.read_bytes())},
              'input': identity(source), 'output': identity(result),
              'alreadyAnnotated': result == source}
    if result != source:
        module.write_bytes(result)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.module), indent=2))


if __name__ == '__main__':
    main()
