"""Verify both shared BF16 policies are linked through the actual mtmd target."""
from __future__ import annotations
import json
from pathlib import Path
import shlex


def runtime_metadata() -> dict:
    return {'version': 1, 'modes': ['legacy-expand', 'retain-checked'], 'default': 'legacy-expand'}


def validate_metadata(info: dict) -> None:
    value = info.get('visionBf16Storage')
    if value is not None and value != runtime_metadata():
        raise ValueError('Unknown vision BF16 runtime capability')
    if info.get('experimentalVisionBf16', False) or any(
            option.startswith(('-DLCB_WEBGPU_BF16_STORAGE=', '-DLCB_WEBGPU_BF16_STORAGE:'))
            for option in info.get('cmakeCommand', [])):
        raise ValueError('Obsolete build-time vision BF16 selection')


def verify(database: Path, *, webgpu: bool) -> None:
    entries = json.loads(database.read_text())
    for filename in ('clip.cpp', 'mtmd.cpp'):
        selected = [entry for entry in entries if Path(entry['file']).name == filename]
        if len(selected) != 1:
            raise ValueError('Expected exactly one ' + filename + ' translation unit')
        arguments = selected[0].get('arguments') or shlex.split(selected[0]['command'])
        normalized = []
        iterator = iter(arguments)
        for arg in iterator:
            normalized.append(arg + next(iterator, '') if arg in ('-D', '-U') else arg)
        if any(arg.startswith(('-DLCB_MTMD_RETAIN_BF16', '-ULCB_MTMD_RETAIN_BF16')) for arg in normalized):
            raise ValueError('Obsolete build-time vision BF16 compile definition')
        definitions = [arg for arg in normalized if arg.startswith(
            ('-DLCB_MTMD_WEBGPU_BF16_RUNTIME', '-ULCB_MTMD_WEBGPU_BF16_RUNTIME'))]
        expected = ['-DLCB_MTMD_WEBGPU_BF16_RUNTIME=' + str(int(webgpu))]
        if definitions != expected:
            raise ValueError(f'Vision BF16 compile definition mismatch: expected {expected}, got {definitions}')
